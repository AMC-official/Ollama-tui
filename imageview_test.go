package main

import (
	"bytes"
	"errors"
	"image"
	"image/color"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func fakeEnv(vars map[string]string) func(string) string {
	return func(k string) string { return vars[k] }
}

func TestDetectImageProtocol(t *testing.T) {
	tests := []struct {
		name string
		env  map[string]string
		want imgProtocol
	}{
		{"nothing set", map[string]string{}, imgNone},
		{"kitty window", map[string]string{"KITTY_WINDOW_ID": "1"}, imgKitty},
		{"kitty term", map[string]string{"TERM": "xterm-kitty"}, imgKitty},
		{"wezterm", map[string]string{"TERM_PROGRAM": "WezTerm"}, imgITerm},
		{"wezterm pane", map[string]string{"WEZTERM_PANE": "3"}, imgITerm},
		{"ghostty", map[string]string{"TERM_PROGRAM": "ghostty"}, imgKitty},
		{"iterm2", map[string]string{"TERM_PROGRAM": "iTerm.app"}, imgITerm},
		{"inside tmux", map[string]string{"KITTY_WINDOW_ID": "1", "TMUX": "/tmp/tmux-1000/default,1,0"}, imgNone},
		{"forced off", map[string]string{"OLLAMA_TUI_IMAGE": "off", "KITTY_WINDOW_ID": "1"}, imgNone},
		{"forced iterm", map[string]string{"OLLAMA_TUI_IMAGE": "iterm"}, imgITerm},
		{"forced kitty in tmux", map[string]string{"OLLAMA_TUI_IMAGE": "kitty", "TMUX": "x"}, imgKitty},
		{"plain xterm", map[string]string{"TERM": "xterm-256color"}, imgNone},
	}
	for _, tc := range tests {
		if got := detectImageProtocol(fakeEnv(tc.env)); got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
}

func TestFitCells(t *testing.T) {
	tests := []struct {
		w, h, maxCols, maxRows int
		cols, rows             int
	}{
		{800, 400, 80, 40, 80, 20}, // wide: fills the width
		{400, 800, 80, 20, 20, 20}, // tall: limited by the height
		{64, 64, 80, 40, 8, 4},     // small pictures are not blown up
		{0, 0, 80, 24, 80, 24},     // unknown size: the whole box
	}
	for _, tc := range tests {
		cols, rows := fitCells(tc.w, tc.h, tc.maxCols, tc.maxRows)
		if cols != tc.cols || rows != tc.rows {
			t.Errorf("fitCells(%d,%d,%d,%d) = %d x %d, want %d x %d", tc.w, tc.h, tc.maxCols, tc.maxRows, cols, rows, tc.cols, tc.rows)
		}
	}
}

func TestKittySequence(t *testing.T) {
	one := kittySequence([]byte("hi"), 3, 2)
	if one != "\x1b_Ga=T,f=100,q=2,c=3,r=2,m=0;aGk=\x1b\\" {
		t.Errorf("unexpected sequence: %q", one)
	}

	big := kittySequence(bytes.Repeat([]byte{7}, 5000), 10, 5) // 6668 base64 characters: two pieces
	if n := strings.Count(big, "\x1b_G"); n != 2 {
		t.Errorf("expected 2 pieces, got %d", n)
	}
	if !strings.HasPrefix(big, "\x1b_Ga=T,f=100,q=2,c=10,r=5,m=1;") {
		t.Errorf("the first piece carries the settings and says more follows: %q", big[:40])
	}
	if !strings.Contains(big, "\x1b_Gq=2,m=0;") {
		t.Error("the last piece must say that nothing follows")
	}
}

func TestItermSequence(t *testing.T) {
	want := "\x1b]1337;File=inline=1;size=2;width=4;height=3;preserveAspectRatio=1:aGk=\a"
	if got := itermSequence([]byte("hi"), 4, 3); got != want {
		t.Errorf("got %q", got)
	}
	seq, cols, rows := imageSequence(imgITerm, preparedImage{data: []byte("hi"), width: 800, height: 400}, 80, 40)
	if cols != 80 || rows != 20 || !strings.HasPrefix(seq, "\x1b]1337;") {
		t.Errorf("unexpected result: %q %d %d", seq, cols, rows)
	}
	if seq, _, _ := imageSequence(imgKitty, preparedImage{data: []byte("hi")}, 10, 5); !strings.HasPrefix(seq, "\x1b_G") {
		t.Errorf("kitty should get a kitty sequence: %q", seq)
	}
}

func TestScaleDown(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 40, 20))
	for y := 0; y < 20; y++ {
		for x := 0; x < 40; x++ {
			src.SetRGBA(x, y, color.RGBA{R: 200, G: 100, B: 50, A: 255})
		}
	}
	dst := scaleDown(src, 10)
	if b := dst.Bounds(); b.Dx() != 10 || b.Dy() != 5 {
		t.Fatalf("expected 10 x 5, got %d x %d", b.Dx(), b.Dy())
	}
	r, g, b, a := dst.At(3, 2).RGBA()
	if r>>8 != 200 || g>>8 != 100 || b>>8 != 50 || a>>8 != 255 {
		t.Errorf("a flat colour must stay the same: %d %d %d %d", r>>8, g>>8, b>>8, a>>8)
	}
	if same := scaleDown(src, 100); same != image.Image(src) {
		t.Error("a small picture should be returned as it is")
	}
}

func writeTestPNG(t *testing.T, path string, w, h int) {
	t.Helper()
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			img.SetNRGBA(x, y, color.NRGBA{R: uint8(x), G: uint8(y), B: 100, A: 255})
		}
	}
	var buf bytes.Buffer
	if err := png.Encode(&buf, img); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
}

func TestPrepareImage(t *testing.T) {
	dir := t.TempDir()

	pngPath := filepath.Join(dir, "a.png")
	writeTestPNG(t, pngPath, 30, 10)
	p, err := prepareImage(pngPath, imgKitty)
	if err != nil || p.width != 30 || p.height != 10 || !isPNG(p.data) {
		t.Errorf("a small PNG should pass: %+v %v", p, err)
	}

	// A JPEG is converted to PNG, because kitty only takes PNG.
	jpgPath := filepath.Join(dir, "b.jpg")
	var buf bytes.Buffer
	if err := jpeg.Encode(&buf, image.NewRGBA(image.Rect(0, 0, 20, 20)), nil); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(jpgPath, buf.Bytes(), 0644); err != nil {
		t.Fatal(err)
	}
	p, err = prepareImage(jpgPath, imgKitty)
	if err != nil || p.width != 20 || !isPNG(p.data) {
		t.Errorf("a JPEG should become a PNG: %+v %v", p, err)
	}

	// A big picture is shrunk.
	bigPath := filepath.Join(dir, "big.png")
	writeTestPNG(t, bigPath, maxImageSide+200, 100)
	p, err = prepareImage(bigPath, imgKitty)
	if err != nil || p.width != maxImageSide {
		t.Errorf("a big picture should be shrunk to %d pixels: %+v %v", maxImageSide, p, err)
	}
}

func TestPrepareImageUnknownFormatAndErrors(t *testing.T) {
	dir := t.TempDir()
	odd := filepath.Join(dir, "x.webp")
	if err := os.WriteFile(odd, []byte("RIFF....WEBPVP8 not really a picture"), 0644); err != nil {
		t.Fatal(err)
	}
	if _, err := prepareImage(odd, imgKitty); !errors.Is(err, errNoInlineFormat) {
		t.Errorf("kitty cannot draw it: %v", err)
	}
	p, err := prepareImage(odd, imgITerm)
	if err != nil || p.width != 0 || !strings.HasPrefix(string(p.data), "RIFF") {
		t.Errorf("iTerm2-style terminals get the original file: %+v %v", p, err)
	}
	if _, err := prepareImage(dir, imgKitty); err == nil {
		t.Error("a folder is not a picture")
	}
	if _, err := prepareImage(filepath.Join(dir, "missing.png"), imgKitty); err == nil {
		t.Error("a missing file should fail")
	}
}

func TestExternalViewerCommand(t *testing.T) {
	has := func(names ...string) func(string) (string, error) {
		return func(name string) (string, error) {
			for _, n := range names {
				if n == name {
					return "/usr/bin/" + name, nil
				}
			}
			return "", errors.New("not found")
		}
	}
	x11 := fakeEnv(map[string]string{"DISPLAY": ":0"})

	if name, args, err := externalViewerCommand("darwin", has(), fakeEnv(nil), "/p.png"); err != nil || name != "open" || args[0] != "/p.png" {
		t.Errorf("macOS: %s %v %v", name, args, err)
	}
	if name, args, err := externalViewerCommand("windows", has(), fakeEnv(nil), `C:\p.png`); err != nil || name != "cmd" || args[0] != "/c" {
		t.Errorf("Windows: %s %v %v", name, args, err)
	}
	if name, args, err := externalViewerCommand("freebsd", has("xdg-open"), x11, "/p.png"); err != nil || name != "xdg-open" || len(args) != 1 {
		t.Errorf("xdg-open: %s %v %v", name, args, err)
	}
	if name, args, err := externalViewerCommand("linux", has("gio"), x11, "/p.png"); err != nil || name != "gio" || len(args) != 2 || args[0] != "open" || args[1] != "/p.png" {
		t.Errorf("gio: %s %v %v", name, args, err)
	}
	if _, _, err := externalViewerCommand("linux", has(), x11, "/p.png"); err == nil || !strings.Contains(err.Error(), "xdg-utils") {
		t.Errorf("a missing viewer should say what to install: %v", err)
	}
	if _, _, err := externalViewerCommand("linux", has("xdg-open"), fakeEnv(nil), "/p.png"); err == nil || !strings.Contains(err.Error(), "graphical") {
		t.Errorf("without a display there is nothing to open: %v", err)
	}
	termux := fakeEnv(map[string]string{"TERMUX_VERSION": "0.118"})
	if name, _, err := externalViewerCommand("linux", has("termux-open"), termux, "/p.png"); err != nil || name != "termux-open" {
		t.Errorf("Termux: %s %v", name, err)
	}
}

func TestImageShowCmdRun(t *testing.T) {
	var out bytes.Buffer
	c := &imageShowCmd{seq: "\x1b_Ga=T;AAAA\x1b\\", caption: "cat.png • test"}
	c.SetStdin(strings.NewReader("\n"))
	c.SetStdout(&out)
	c.SetStderr(&out)
	if err := c.Run(); err != nil {
		t.Fatal(err)
	}
	got := out.String()
	for _, want := range []string{"cat.png • test", "\x1b_Ga=T;AAAA", "Press Enter", kittyDeleteAll, "\x1b[?1049l"} {
		if !strings.Contains(got, want) {
			t.Errorf("the output should contain %q: %q", want, got)
		}
	}
}
