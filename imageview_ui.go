package main

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
)

// --- Image viewer, part 2: how it hooks into the app ---
//
// Flow: viewImage -> prepareImageCmd (decodes and shrinks the picture in the
// background) -> imageReadyMsg -> tea.Exec(imageShowCmd) which hands the whole
// terminal over for a moment, draws the picture and waits for Enter.
// Without inline image support (or for formats the terminal cannot draw) the
// picture is opened in the default viewer of the system instead.

// imageNotePrefix starts the grey note that is added to the chat when an image
// is sent ("🖼️ Image: photo.png"). main.go writes the same text.
const imageNotePrefix = "\U0001F5BC\uFE0F Image: "

const maxSeenImages = 50

// imageRef points at a picture by name and by where it was on disk.
type imageRef struct {
	Name string
	Path string
}

// imageReadyMsg: the picture is prepared and can be drawn.
type imageReadyMsg struct {
	ref     imageRef
	seq     string // the escape code that draws it
	caption string
}

// imageFallbackMsg: the picture cannot be drawn inline, open it elsewhere.
type imageFallbackMsg struct{ ref imageRef }

// imageDoneMsg reports how showing a picture went.
type imageDoneMsg struct {
	ref imageRef
	how string // "inline" or "viewer"
	err error
}

// --- Finding the pictures of the chat ---

// rememberImages keeps where the images of a sent message came from, so they
// can be shown again later (the message itself only keeps a note).
func (m *model) rememberImages(atts []Attachment) {
	for _, a := range atts {
		if a.IsImage {
			m.seenImages = append(m.seenImages, imageRef{Name: a.Name, Path: a.Path})
		}
	}
	if len(m.seenImages) > maxSeenImages {
		m.seenImages = append([]imageRef(nil), m.seenImages[len(m.seenImages)-maxSeenImages:]...)
	}
}

// imageFromNote finds the picture behind a chat note, if the note is an image
// note. The path comes from rememberImages (newest match wins).
func (m model) imageFromNote(content string) (imageRef, bool) {
	if !strings.HasPrefix(content, imageNotePrefix) {
		return imageRef{}, false
	}
	name := strings.TrimPrefix(content, imageNotePrefix)
	for i := len(m.seenImages) - 1; i >= 0; i-- {
		if m.seenImages[i].Name == name {
			return m.seenImages[i], true
		}
	}
	return imageRef{Name: name}, true // a note from a saved chat: the path is unknown
}

// chatImages lists the pictures of this chat (oldest first), followed by the
// images that are attached but not sent yet.
func (m model) chatImages() []imageRef {
	var out []imageRef
	for _, msg := range m.messages {
		if msg.Role != "System" {
			continue
		}
		if ref, ok := m.imageFromNote(msg.Content); ok {
			out = append(out, ref)
		}
	}
	for _, a := range m.attachments {
		if a.IsImage {
			out = append(out, imageRef{Name: a.Name, Path: a.Path})
		}
	}
	return out
}

// imageForMessage is the picture to show for the selected message in List
// Mode: the one the message is a note for, otherwise the newest picture.
func (m model) imageForMessage(i int) (imageRef, bool) {
	if i >= 0 && i < len(m.messages) && m.messages[i].Role == "System" {
		if ref, ok := m.imageFromNote(m.messages[i].Content); ok {
			return ref, true
		}
	}
	all := m.chatImages()
	if len(all) == 0 {
		return imageRef{}, false
	}
	return all[len(all)-1], true
}

// --- Showing a picture ---

// viewImage starts showing a picture the best way the terminal allows.
func (m model) viewImage(ref imageRef) tea.Cmd {
	if ref.Path == "" {
		return func() tea.Msg {
			return imageDoneMsg{ref: ref, how: "viewer", err: errors.New("the file of this picture is not known (the chat was loaded from a saved file)")}
		}
	}
	proto := detectImageProtocol(os.Getenv)
	if proto == imgNone {
		return openExternalCmd(ref)
	}
	cols, rows := m.width-2, m.height-6 // room for the caption and the prompt
	if cols < 10 {
		cols = 10
	}
	if rows < 4 {
		rows = 4
	}
	return prepareImageCmd(ref, proto, cols, rows)
}

// prepareImageCmd decodes and shrinks the picture in the background.
func prepareImageCmd(ref imageRef, proto imgProtocol, maxCols, maxRows int) tea.Cmd {
	return func() tea.Msg {
		p, err := prepareImage(ref.Path, proto)
		if errors.Is(err, errNoInlineFormat) {
			return imageFallbackMsg{ref: ref}
		}
		if err != nil {
			return imageDoneMsg{ref: ref, how: "inline", err: err}
		}
		seq, cols, rows := imageSequence(proto, p, maxCols, maxRows)
		size := ""
		if p.width > 0 {
			size = fmt.Sprintf(" • %dx%d px", p.width, p.height)
		}
		caption := fmt.Sprintf("%s%s • %d x %d cells • %s", ref.Name, size, cols, rows, proto)
		return imageReadyMsg{ref: ref, seq: seq, caption: caption}
	}
}

// imageShowCmd draws a picture while Bubble Tea has let go of the terminal
// (tea.Exec). It uses a screen of its own, so the chat comes back untouched.
type imageShowCmd struct {
	seq     string
	caption string
	stdin   io.Reader
	stdout  io.Writer
	stderr  io.Writer
}

func (c *imageShowCmd) SetStdin(r io.Reader)  { c.stdin = r }
func (c *imageShowCmd) SetStdout(w io.Writer) { c.stdout = w }
func (c *imageShowCmd) SetStderr(w io.Writer) { c.stderr = w }

func (c *imageShowCmd) Run() error {
	var out io.Writer = os.Stdout
	if c.stdout != nil {
		out = c.stdout
	}
	var in io.Reader = os.Stdin
	if c.stdin != nil {
		in = c.stdin
	}
	fmt.Fprint(out, "\x1b[?1049h\x1b[2J\x1b[H") // a fresh screen
	fmt.Fprint(out, c.caption+"\r\n\r\n")
	fmt.Fprint(out, c.seq)
	fmt.Fprint(out, "\r\n\r\nPress Enter to go back...")
	_, _ = bufio.NewReader(in).ReadString('\n')
	if strings.HasPrefix(c.seq, "\x1b_G") {
		fmt.Fprint(out, kittyDeleteAll) // tell kitty to forget the picture
	}
	fmt.Fprint(out, "\x1b[?1049l") // back to the screen we came from
	return nil
}

// externalViewerCommand picks the command that opens a file in the default
// viewer. The system is passed in, so this can be tested.
func externalViewerCommand(goos string, lookPath func(string) (string, error), getenv func(string) string, path string) (string, []string, error) {
	switch goos {
	case "darwin":
		return "open", []string{path}, nil
	case "windows":
		return "cmd", []string{"/c", "start", "", path}, nil
	}
	// Linux and the BSDs. Termux and WSL have their own openers.
	if getenv("TERMUX_VERSION") != "" {
		if _, err := lookPath("termux-open"); err == nil {
			return "termux-open", []string{path}, nil
		}
	}
	if getenv("WSL_DISTRO_NAME") != "" {
		if _, err := lookPath("wslview"); err == nil {
			return "wslview", []string{path}, nil
		}
	}
	if getenv("DISPLAY") == "" && getenv("WAYLAND_DISPLAY") == "" {
		return "", nil, errors.New("no graphical session ($DISPLAY or $WAYLAND_DISPLAY is empty); use a terminal that can show images, such as kitty or WezTerm")
	}
	for _, c := range [][]string{{"xdg-open"}, {"gio", "open"}} {
		if _, err := lookPath(c[0]); err == nil {
			return c[0], append(c[1:len(c):len(c)], path), nil
		}
	}
	return "", nil, errors.New("no image viewer found: install xdg-utils (FreeBSD: pkg install xdg-utils)")
}

// openExternalCmd opens the picture in the default viewer and does not wait for it.
func openExternalCmd(ref imageRef) tea.Cmd {
	return func() tea.Msg {
		if _, err := os.Stat(ref.Path); err != nil {
			return imageDoneMsg{ref: ref, how: "viewer", err: err}
		}
		name, args, err := externalViewerCommand(runtime.GOOS, exec.LookPath, os.Getenv, ref.Path)
		if err != nil {
			return imageDoneMsg{ref: ref, how: "viewer", err: err}
		}
		c := exec.Command(name, args...) // no stdin/stdout set: they go to the null device
		if err := c.Start(); err != nil {
			return imageDoneMsg{ref: ref, how: "viewer", err: err}
		}
		go func() { _ = c.Wait() }() // collect the process when the viewer is closed
		return imageDoneMsg{ref: ref, how: "viewer"}
	}
}

// onImageMsg handles the three messages of the image viewer.
func (m *model) onImageMsg(msg tea.Msg) tea.Cmd {
	switch msg := msg.(type) {
	case imageReadyMsg:
		return tea.Exec(&imageShowCmd{seq: msg.seq, caption: msg.caption}, func(err error) tea.Msg {
			return imageDoneMsg{ref: msg.ref, how: "inline", err: err}
		})

	case imageFallbackMsg:
		return openExternalCmd(msg.ref)

	case imageDoneMsg:
		text := ""
		switch {
		case msg.err != nil:
			text = "Could not show " + msg.ref.Name + ": " + msg.err.Error()
		case msg.how == "viewer":
			text = "Opened " + msg.ref.Name + " in your default image viewer."
		default:
			text = "Showed " + msg.ref.Name + "."
		}
		if m.console {
			mark := "✓ "
			if msg.err != nil {
				mark = "✗ "
			}
			m.consolePrint(mark + text)
			return nil
		}
		m.statusMsg = text
		m.statusIsError = msg.err != nil
		return clearStatusCmd(6 * time.Second)
	}
	return nil
}

// cleanImagePath turns what the user typed after :view into a path.
func cleanImagePath(arg string) string {
	arg = strings.Trim(strings.TrimSpace(arg), `"'`)
	arg = strings.ReplaceAll(arg, `\ `, " ")
	if strings.HasPrefix(arg, "~/") {
		if home, err := os.UserHomeDir(); err == nil {
			arg = filepath.Join(home, arg[2:])
		}
	}
	return filepath.Clean(arg)
}

// isImagePath says whether a file name looks like a picture we can show.
func isImagePath(path string) bool {
	switch strings.ToLower(filepath.Ext(path)) {
	case ".png", ".jpg", ".jpeg", ".webp", ".gif":
		return true
	}
	return false
}
