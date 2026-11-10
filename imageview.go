package main

import (
	"bytes"
	"encoding/base64"
	"errors"
	"fmt"
	"image"
	_ "image/gif"  // lets image.Decode read GIF
	_ "image/jpeg" // lets image.Decode read JPEG
	"image/png"    // PNG: decoding and the re-encoding we send to the terminal
	"math"
	"os"
	"strings"
)

// --- Image viewer, part 1: the parts that need no screen ---
//
// Two families of terminals can draw pictures inline:
//
//   - the kitty graphics protocol: kitty, Ghostty
//   - the iTerm2 inline image protocol: WezTerm, iTerm2, mintty
//
// Everywhere else the picture is opened in the system's default viewer (see
// imageview_ui.go). Set OLLAMA_TUI_IMAGE to kitty, iterm or off to override
// the detection.

type imgProtocol int

const (
	imgNone  imgProtocol = iota // no inline images: use the default viewer
	imgKitty                    // kitty graphics protocol
	imgITerm                    // iTerm2 inline images (also understood by WezTerm)
)

func (p imgProtocol) String() string {
	switch p {
	case imgKitty:
		return "kitty graphics"
	case imgITerm:
		return "iTerm2 inline images"
	}
	return "default viewer"
}

const (
	maxImageFile   = 25 << 20  // bigger files go to the default viewer
	maxImagePixels = 100000000 // refuse to decode anything above 100 megapixels
	maxImageSide   = 1800      // pictures are shrunk to this before they are sent
	kittyDeleteAll = "\x1b_Ga=d,d=A,q=2\x1b\\"
)

// errNoInlineFormat means the picture cannot be drawn inline (for example a
// WebP in a kitty terminal), so the default viewer is used instead.
var errNoInlineFormat = errors.New("this image format cannot be shown inline in this terminal")

// detectImageProtocol looks at the environment variables terminals set. It
// takes getenv as an argument so it can be tested.
func detectImageProtocol(getenv func(string) string) imgProtocol {
	switch strings.ToLower(strings.TrimSpace(getenv("OLLAMA_TUI_IMAGE"))) {
	case "kitty":
		return imgKitty
	case "iterm", "iterm2", "wezterm":
		return imgITerm
	case "off", "none", "external", "viewer":
		return imgNone
	}
	// Inside tmux or screen the escape codes do not get through.
	if getenv("TMUX") != "" || getenv("STY") != "" {
		return imgNone
	}
	term, prog := getenv("TERM"), getenv("TERM_PROGRAM")
	switch {
	case prog == "WezTerm" || getenv("WEZTERM_EXECUTABLE") != "" || getenv("WEZTERM_PANE") != "":
		return imgITerm
	case getenv("KITTY_WINDOW_ID") != "" || term == "xterm-kitty":
		return imgKitty
	case prog == "ghostty" || term == "xterm-ghostty" || getenv("GHOSTTY_RESOURCES_DIR") != "":
		return imgKitty
	case prog == "iTerm.app" || prog == "mintty":
		return imgITerm
	}
	return imgNone
}

// fitCells says how many columns and rows a picture of w x h pixels takes in a
// box of maxCols x maxRows cells. A cell is about twice as tall as it is wide.
// Small pictures are not blown up: eight pixels count as one column.
func fitCells(w, h, maxCols, maxRows int) (cols, rows int) {
	if maxCols < 1 {
		maxCols = 1
	}
	if maxRows < 1 {
		maxRows = 1
	}
	if w <= 0 || h <= 0 {
		return maxCols, maxRows
	}
	if natural := w / 8; natural >= 8 && natural < maxCols {
		maxCols = natural
	}
	cols = maxCols
	rows = int(math.Round(float64(cols) * float64(h) / float64(w) / 2))
	if rows > maxRows {
		rows = maxRows
		cols = int(math.Round(float64(rows) * 2 * float64(w) / float64(h)))
	}
	if cols < 1 {
		cols = 1
	}
	if rows < 1 {
		rows = 1
	}
	return cols, rows
}

// preparedImage is a picture ready to be sent to the terminal.
type preparedImage struct {
	data          []byte // PNG, or the original file when only iTerm2-style terminals can decode it
	width, height int    // in pixels; 0 when unknown
}

func isPNG(data []byte) bool {
	return bytes.HasPrefix(data, []byte("\x89PNG\r\n\x1a\n"))
}

// prepareImage reads the picture and turns it into something the terminal
// understands: a PNG that is not larger than maxImageSide pixels.
func prepareImage(path string, proto imgProtocol) (preparedImage, error) {
	st, err := os.Stat(path)
	if err != nil {
		return preparedImage{}, err
	}
	if st.IsDir() {
		return preparedImage{}, errors.New("this is a folder, not a picture")
	}
	if st.Size() > maxImageFile {
		return preparedImage{}, errNoInlineFormat
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return preparedImage{}, err
	}

	cfg, _, err := image.DecodeConfig(bytes.NewReader(raw))
	if err != nil {
		// WebP has no decoder in the standard library. iTerm2-style terminals
		// decode it themselves, so the original file is sent as it is.
		if proto == imgITerm {
			return preparedImage{data: raw}, nil
		}
		return preparedImage{}, errNoInlineFormat
	}
	if cfg.Width*cfg.Height > maxImagePixels {
		return preparedImage{}, errNoInlineFormat
	}

	img, _, err := image.Decode(bytes.NewReader(raw))
	if err != nil {
		return preparedImage{}, err
	}
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= maxImageSide && h <= maxImageSide && isPNG(raw) {
		return preparedImage{data: raw, width: w, height: h}, nil // already what we need
	}
	if w > maxImageSide || h > maxImageSide {
		img = scaleDown(img, maxImageSide)
		b = img.Bounds()
		w, h = b.Dx(), b.Dy()
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		return preparedImage{}, err
	}
	return preparedImage{data: buf.Bytes(), width: w, height: h}, nil
}

// scaleDown shrinks a picture so that its longest side is maxSide pixels,
// averaging the pixels that fall into each new pixel (so lines stay smooth).
func scaleDown(src image.Image, maxSide int) image.Image {
	b := src.Bounds()
	w, h := b.Dx(), b.Dy()
	longest := w
	if h > longest {
		longest = h
	}
	if longest <= maxSide || maxSide < 1 {
		return src
	}
	scale := float64(maxSide) / float64(longest)
	nw := int(math.Round(float64(w) * scale))
	nh := int(math.Round(float64(h) * scale))
	if nw < 1 {
		nw = 1
	}
	if nh < 1 {
		nh = 1
	}

	dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
	for y := 0; y < nh; y++ {
		y0 := b.Min.Y + y*h/nh
		y1 := b.Min.Y + (y+1)*h/nh
		if y1 <= y0 {
			y1 = y0 + 1
		}
		for x := 0; x < nw; x++ {
			x0 := b.Min.X + x*w/nw
			x1 := b.Min.X + (x+1)*w/nw
			if x1 <= x0 {
				x1 = x0 + 1
			}
			var sr, sg, sb, sa, n uint64
			for yy := y0; yy < y1; yy++ {
				for xx := x0; xx < x1; xx++ {
					r, g, bl, a := src.At(xx, yy).RGBA()
					sr += uint64(r)
					sg += uint64(g)
					sb += uint64(bl)
					sa += uint64(a)
					n++
				}
			}
			off := dst.PixOffset(x, y)
			dst.Pix[off+0] = uint8((sr / n) >> 8)
			dst.Pix[off+1] = uint8((sg / n) >> 8)
			dst.Pix[off+2] = uint8((sb / n) >> 8)
			dst.Pix[off+3] = uint8((sa / n) >> 8)
		}
	}
	return dst
}

// kittySequence is the escape code that makes kitty (or Ghostty) draw a PNG
// over cols x rows cells. The data goes in pieces of 4096 characters, as the
// protocol wants; q=2 keeps the terminal from answering on our input.
func kittySequence(pngData []byte, cols, rows int) string {
	enc := base64.StdEncoding.EncodeToString(pngData)
	var b strings.Builder
	first := true
	for len(enc) > 0 {
		n := 4096
		if len(enc) < n {
			n = len(enc)
		}
		chunk := enc[:n]
		enc = enc[n:]
		more := 0
		if len(enc) > 0 {
			more = 1
		}
		if first {
			fmt.Fprintf(&b, "\x1b_Ga=T,f=100,q=2,c=%d,r=%d,m=%d;%s\x1b\\", cols, rows, more, chunk)
			first = false
		} else {
			fmt.Fprintf(&b, "\x1b_Gq=2,m=%d;%s\x1b\\", more, chunk)
		}
	}
	return b.String()
}

// itermSequence is the escape code for iTerm2-style inline images (WezTerm,
// iTerm2, mintty). The picture is fitted into the box, keeping its proportions.
func itermSequence(data []byte, cols, rows int) string {
	enc := base64.StdEncoding.EncodeToString(data)
	return fmt.Sprintf("\x1b]1337;File=inline=1;size=%d;width=%d;height=%d;preserveAspectRatio=1:%s\a", len(data), cols, rows, enc)
}

// imageSequence builds the escape code for a prepared picture and reports
// the size in cells it will take.
func imageSequence(proto imgProtocol, p preparedImage, maxCols, maxRows int) (seq string, cols, rows int) {
	cols, rows = fitCells(p.width, p.height, maxCols, maxRows)
	if proto == imgKitty {
		return kittySequence(p.data, cols, rows), cols, rows
	}
	return itermSequence(p.data, cols, rows), cols, rows
}
