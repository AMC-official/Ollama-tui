package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// --- Model manager (pull / delete) ---

// validModelPart accepts what Ollama model names and tags are made of, so we
// never put odd characters into a URL.
func validModelPart(s string, allowSlash bool) bool {
	if s == "" {
		return false
	}
	for i, r := range s {
		ok := (r >= 'a' && r <= 'z') || (r >= '0' && r <= '9')
		if !ok && i > 0 {
			ok = r == '.' || r == '_' || r == '-' || (allowSlash && r == '/')
		}
		if !ok {
			return false
		}
	}
	return true
}

func friendlyPullStatus(s string) string {
	switch {
	case strings.HasPrefix(s, "pulling manifest"):
		return "Fetching the model manifest..."
	case strings.HasPrefix(s, "pulling"):
		return "Downloading..."
	case strings.HasPrefix(s, "verifying"):
		return "Verifying the download..."
	case strings.HasPrefix(s, "writing"):
		return "Writing the manifest..."
	case s == "success":
		return "Done!"
	}
	return s
}

func (m *model) openPull() {
	for i := range m.pullInputs {
		m.pullInputs[i].SetValue("")
	}
	m.pullStep = pullInput
	m.pullFocus = 0
	m.pullErr = ""
	m.focusPull()
	m.state = stateModelPull
}

// focusPull gives the keyboard to the field m.pullFocus points at.
func (m *model) focusPull() {
	for i := range m.pullInputs {
		m.pullInputs[i].Blur()
	}
	m.pullInputs[m.pullFocus].Focus()
}

func (m model) pullScreen() string {
	gray := lipgloss.NewStyle().Foreground(colorGray)
	rows := []string{aiLabelStyle.Render("Pull a model"), ""}

	switch m.pullStep {
	case pullInput:
		label := func(i int, name string) string {
			name = fmt.Sprintf("%-18s", name)
			if m.pullFocus == i {
				return aiLabelStyle.Render(name)
			}
			return gray.Render(name)
		}
		rows = append(rows,
			lipgloss.JoinHorizontal(lipgloss.Top, label(0, "Model name"), m.pullInputs[0].View()),
			lipgloss.JoinHorizontal(lipgloss.Top, label(1, "Parameters / tag"), m.pullInputs[1].View()),
			"")
		if m.pullErr != "" {
			rows = append(rows, lipgloss.NewStyle().Foreground(colorRed).Render("✗ "+m.pullErr), "")
		}
		rows = append(rows, gray.Render("Example: gemma3 + 4b gives gemma3:4b. Leave the second field empty for the default (latest).\nTab: switch field • Enter: check the library • Esc: back"))
	case pullChecking:
		rows = append(rows, "Checking "+m.pullModel+" in the Ollama library...", "", gray.Render("Esc: cancel"))
	case pullConfirm:
		size := ""
		if m.pullSize > 0 {
			size = " (" + humanBytes(m.pullSize) + ")"
		}
		rows = append(rows, lipgloss.NewStyle().Foreground(colorGreen).Render("✓ "+m.pullModel+" found"+size), "", "Download it? (y/n)")
	case pullDownloading:
		frac := 0.0
		if m.pullTotal > 0 {
			frac = float64(m.pullDone) / float64(m.pullTotal)
		}
		barW := m.width - 24
		if barW > 50 {
			barW = 50
		}
		if barW < 10 {
			barW = 10
		}
		line := fmt.Sprintf("%s / %s", humanBytes(m.pullDone), humanBytes(m.pullTotal))
		if secs := time.Since(m.pullStart).Seconds(); secs >= 2 && m.pullDone > 0 {
			speed := float64(m.pullDone) / secs
			line += fmt.Sprintf(" • %s/s", humanBytes(int64(speed)))
			if m.pullTotal > m.pullDone && speed > 0 {
				eta := time.Duration(float64(m.pullTotal-m.pullDone)/speed) * time.Second
				line += " • ETA " + formatDuration(eta)
			}
		}
		rows = append(rows,
			"Downloading "+m.pullModel,
			"",
			progressBar(frac, barW)+fmt.Sprintf(" %3.0f%%", frac*100),
			line,
			gray.Render(friendlyPullStatus(m.pullStatus)),
			"",
			gray.Render("Esc: cancel (what is already downloaded is kept)"))
	}

	if sl := m.statusLine(); sl != "" {
		rows = append(rows, "", sl)
	}
	return m.pageStyle().Render(lipgloss.JoinVertical(lipgloss.Left, rows...))
}

// checkRegistry asks the public Ollama library whether name:tag exists and how
// big the download is, without downloading anything.
func checkRegistry(ref string) tea.Cmd {
	return func() tea.Msg {
		name, tag := ref, "latest"
		if i := strings.LastIndex(ref, ":"); i >= 0 {
			name, tag = ref[:i], ref[i+1:]
		}
		repo := name
		if !strings.Contains(repo, "/") {
			repo = "library/" + repo
		}
		req, err := http.NewRequest("GET", "https://registry.ollama.ai/v2/"+repo+"/manifests/"+tag, nil)
		if err != nil {
			return pullCheckedMsg{ref: ref, err: err}
		}
		req.Header.Set("Accept", "application/vnd.docker.distribution.manifest.v2+json")
		resp, err := apiClient.Do(req)
		if err != nil {
			return pullCheckedMsg{ref: ref, err: fmt.Errorf("could not reach registry.ollama.ai: %v", err)}
		}
		defer resp.Body.Close()

		switch resp.StatusCode {
		case http.StatusOK:
		case http.StatusNotFound:
			return pullCheckedMsg{ref: ref, notFound: true}
		default:
			return pullCheckedMsg{ref: ref, err: fmt.Errorf("the library answered %s", resp.Status)}
		}

		var man struct {
			Config struct {
				Size int64 `json:"size"`
			} `json:"config"`
			Layers []struct {
				Size int64 `json:"size"`
			} `json:"layers"`
		}
		if err := json.NewDecoder(resp.Body).Decode(&man); err != nil {
			return pullCheckedMsg{ref: ref, err: fmt.Errorf("unexpected reply from the library: %v", err)}
		}
		total := man.Config.Size
		for _, l := range man.Layers {
			total += l.Size
		}
		return pullCheckedMsg{ref: ref, size: total}
	}
}

func waitForPull(ch chan pullEvent) tea.Cmd {
	return func() tea.Msg {
		ev, ok := <-ch
		if !ok {
			return pullDoneMsg{ch: ch}
		}
		return pullProgressMsg{ch: ch, ev: ev}
	}
}

// startPull streams the progress of /api/pull into ch and closes it when done.
// totalHint is the full download size from the library check (0 if unknown).
func startPull(ctx context.Context, host, ref string, totalHint int64, ch chan pullEvent) tea.Cmd {
	return func() tea.Msg {
		defer close(ch)

		send := func(ev pullEvent) bool {
			select {
			case ch <- ev:
				return true
			case <-ctx.Done():
				return false
			}
		}

		body, _ := json.Marshal(map[string]interface{}{"model": ref, "name": ref, "stream": true})
		req, err := http.NewRequestWithContext(ctx, "POST", host+"/api/pull", bytes.NewReader(body))
		if err != nil {
			send(pullEvent{Err: err.Error()})
			return nil
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := http.DefaultClient.Do(req)
		if err != nil {
			if ctx.Err() == nil {
				send(pullEvent{Err: fmt.Sprintf("failed to reach %s: %v", host, err)})
			}
			return nil
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
			detail := strings.TrimSpace(string(b))
			var e struct {
				Error string `json:"error"`
			}
			if json.Unmarshal(b, &e) == nil && e.Error != "" {
				detail = e.Error
			}
			send(pullEvent{Err: fmt.Sprintf("Ollama answered %s: %s", resp.Status, detail)})
			return nil
		}

		// Ollama reports each layer separately; add them up for one progress bar.
		done := map[string]int64{}
		totals := map[string]int64{}
		scanner := bufio.NewScanner(resp.Body)
		scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
		for scanner.Scan() {
			var line struct {
				Status    string `json:"status"`
				Digest    string `json:"digest"`
				Total     int64  `json:"total"`
				Completed int64  `json:"completed"`
				Error     string `json:"error"`
			}
			if json.Unmarshal(scanner.Bytes(), &line) != nil {
				continue
			}
			if line.Error != "" {
				send(pullEvent{Err: line.Error})
				return nil
			}
			ev := pullEvent{Status: line.Status}
			if line.Digest != "" && line.Total > 0 {
				done[line.Digest] = line.Completed
				totals[line.Digest] = line.Total
				var d, t int64
				for _, v := range done {
					d += v
				}
				for _, v := range totals {
					t += v
				}
				ev.Completed = d
				ev.Total = t
				if totalHint > t {
					ev.Total = totalHint
				}
			}
			if !send(ev) {
				return nil
			}
		}
		if err := scanner.Err(); err != nil && ctx.Err() == nil {
			send(pullEvent{Err: "connection lost: " + err.Error()})
		}
		return nil
	}
}

func deleteModel(host, name string) tea.Cmd {
	return func() tea.Msg {
		body, _ := json.Marshal(map[string]string{"model": name, "name": name})
		req, err := http.NewRequest(http.MethodDelete, host+"/api/delete", bytes.NewReader(body))
		if err != nil {
			return deleteDoneMsg{name: name, err: err}
		}
		req.Header.Set("Content-Type", "application/json")
		resp, err := apiClient.Do(req)
		if err != nil {
			return deleteDoneMsg{name: name, err: err}
		}
		defer resp.Body.Close()
		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(io.LimitReader(resp.Body, 2048))
			return deleteDoneMsg{name: name, err: fmt.Errorf("ollama answered %s: %s", resp.Status, strings.TrimSpace(string(b)))}
		}
		return deleteDoneMsg{name: name}
	}
}
