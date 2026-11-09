package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strconv"
	"strings"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

// --- System stats: CPU, RAM, swap and GPU (Linux and FreeBSD) ---

// cpuTimes is one reading of the CPU time counters.
type cpuTimes struct{ total, idle uint64 }

type GPUStat struct {
	Name     string
	Util     float64 // percent, -1 if unknown
	MemUsed  uint64  // bytes
	MemTotal uint64
}

type SystemStats struct {
	CPU       float64 // percent, -1 if unknown
	Cores     int
	MemTotal  uint64
	MemUsed   uint64
	SwapTotal uint64
	SwapUsed  uint64
	GPUs      []GPUStat
	Note      string
}

type statsTickMsg struct{ gen int }
type statsMsg struct {
	gen   int
	stats SystemStats
	cpu   cpuTimes
}

func statsTickCmd(gen int) tea.Cmd {
	return tea.Tick(2*time.Second, func(time.Time) tea.Msg { return statsTickMsg{gen: gen} })
}

// collectStatsCmd reads the system load off the UI goroutine.
func collectStatsCmd(prev cpuTimes, gen int) tea.Cmd {
	return func() tea.Msg {
		stats, cur := readSystemStats(prev, true)
		return statsMsg{gen: gen, stats: stats, cpu: cur}
	}
}

// toggleStats shows or hides the percentages in the header. Nothing is read
// (and no timer runs) while they are hidden.
func (m *model) toggleStats() tea.Cmd {
	m.showStats = !m.showStats
	m.statsGen++ // timers of an earlier switch-on stop by themselves
	if !m.showStats {
		m.haveStats = false
		return nil
	}
	return collectStatsCmd(m.cpuPrev, m.statsGen)
}

func runCmd(name string, args ...string) (string, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, name, args...).Output()
	return string(out), err
}

func readUint(path string) (uint64, bool) {
	b, err := os.ReadFile(path)
	if err != nil {
		return 0, false
	}
	n, err := strconv.ParseUint(strings.TrimSpace(string(b)), 10, 64)
	return n, err == nil
}

// readSystemStats reads CPU, memory, swap and (optionally) GPU load. The CPU
// percentage needs the previous reading; the new one is returned for next time.
func readSystemStats(prev cpuTimes, withGPU bool) (SystemStats, cpuTimes) {
	s := SystemStats{CPU: -1, Cores: runtime.NumCPU()}
	var cur cpuTimes

	switch runtime.GOOS {
	case "linux", "android":
		if b, err := os.ReadFile("/proc/stat"); err == nil {
			if t, ok := parseProcStat(string(b)); ok {
				cur = t
			}
		}
		if b, err := os.ReadFile("/proc/meminfo"); err == nil {
			s.MemTotal, s.MemUsed, s.SwapTotal, s.SwapUsed = parseMeminfo(string(b))
		}
	case "freebsd":
		if out, err := runCmd("sysctl", "-n", "kern.cp_time"); err == nil {
			if t, ok := parseCPTime(out); ok {
				cur = t
			}
		}
		if out, err := runCmd("sysctl", "-n", "hw.physmem", "hw.pagesize", "vm.stats.vm.v_free_count", "vm.stats.vm.v_inactive_count"); err == nil {
			s.MemTotal, s.MemUsed = parseFreeBSDMemory(out)
		}
		if out, err := runCmd("swapinfo", "-k"); err == nil {
			s.SwapTotal, s.SwapUsed = parseSwapinfo(out)
		}
	default:
		s.Note = "CPU, RAM and swap numbers are only available on Linux and FreeBSD for now."
	}

	if prev.total > 0 && cur.total > 0 {
		s.CPU = cpuPercent(prev, cur)
	}
	if withGPU {
		s.GPUs = readGPUs()
	}
	return s, cur
}

func readGPUs() []GPUStat {
	var gpus []GPUStat
	if out, err := runCmd("nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"); err == nil {
		gpus = append(gpus, parseNvidiaSmi(out)...)
	}
	if runtime.GOOS == "linux" {
		files, _ := filepath.Glob("/sys/class/drm/card[0-9]*/device/gpu_busy_percent")
		for _, f := range files {
			dir := filepath.Dir(f)
			g := GPUStat{Name: "AMD GPU (" + filepath.Base(filepath.Dir(dir)) + ")", Util: -1}
			if v, ok := readUint(f); ok {
				g.Util = float64(v)
			}
			if v, ok := readUint(filepath.Join(dir, "mem_info_vram_used")); ok {
				g.MemUsed = v
			}
			if v, ok := readUint(filepath.Join(dir, "mem_info_vram_total")); ok {
				g.MemTotal = v
			}
			gpus = append(gpus, g)
		}
	}
	return gpus
}

// parseMeminfo reads /proc/meminfo: total and used RAM, total and used swap (bytes).
func parseMeminfo(text string) (memTotal, memUsed, swapTotal, swapUsed uint64) {
	vals := map[string]uint64{}
	for _, line := range strings.Split(text, "\n") {
		key, rest, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		fields := strings.Fields(rest)
		if len(fields) == 0 {
			continue
		}
		n, err := strconv.ParseUint(fields[0], 10, 64)
		if err != nil {
			continue
		}
		vals[key] = n * 1024 // the values are in kB
	}
	memTotal = vals["MemTotal"]
	avail, ok := vals["MemAvailable"]
	if !ok {
		avail = vals["MemFree"] + vals["Buffers"] + vals["Cached"]
	}
	if memTotal > avail {
		memUsed = memTotal - avail
	}
	swapTotal = vals["SwapTotal"]
	if swapTotal > vals["SwapFree"] {
		swapUsed = swapTotal - vals["SwapFree"]
	}
	return memTotal, memUsed, swapTotal, swapUsed
}

func sumUint(fields []string) (total uint64, ok bool) {
	for _, f := range fields {
		n, err := strconv.ParseUint(f, 10, 64)
		if err != nil {
			return 0, false
		}
		total += n
	}
	return total, true
}

// parseProcStat reads the first line of /proc/stat ("cpu  user nice system idle iowait ...").
func parseProcStat(text string) (cpuTimes, bool) {
	for _, line := range strings.Split(text, "\n") {
		fields := strings.Fields(line)
		if len(fields) < 6 || fields[0] != "cpu" {
			continue
		}
		nums := fields[1:]
		if len(nums) > 8 {
			nums = nums[:8] // guest time is already part of user time
		}
		total, ok := sumUint(nums)
		idle, ok2 := sumUint(nums[3:5]) // idle + iowait
		if !ok || !ok2 {
			return cpuTimes{}, false
		}
		return cpuTimes{total: total, idle: idle}, true
	}
	return cpuTimes{}, false
}

// parseCPTime reads FreeBSD's kern.cp_time: "user nice sys intr idle".
func parseCPTime(out string) (cpuTimes, bool) {
	fields := strings.Fields(out)
	if len(fields) < 5 {
		return cpuTimes{}, false
	}
	total, ok := sumUint(fields[:5])
	idle, ok2 := sumUint(fields[4:5])
	if !ok || !ok2 {
		return cpuTimes{}, false
	}
	return cpuTimes{total: total, idle: idle}, true
}

// cpuPercent is how busy the CPU was between two readings (-1 if it cannot be told).
func cpuPercent(prev, cur cpuTimes) float64 {
	if cur.total <= prev.total || cur.idle < prev.idle {
		return -1
	}
	dt := float64(cur.total - prev.total)
	di := float64(cur.idle - prev.idle)
	p := (1 - di/dt) * 100
	if p < 0 {
		p = 0
	}
	if p > 100 {
		p = 100
	}
	return p
}

// parseFreeBSDMemory reads four numbers: physmem, page size, free pages, inactive pages.
func parseFreeBSDMemory(out string) (total, used uint64) {
	f := strings.Fields(out)
	if len(f) < 4 {
		return 0, 0
	}
	n := make([]uint64, 4)
	for i := range n {
		v, err := strconv.ParseUint(f[i], 10, 64)
		if err != nil {
			return 0, 0
		}
		n[i] = v
	}
	total = n[0]
	reclaimable := (n[2] + n[3]) * n[1] // free and inactive pages can be used right away
	if total > reclaimable {
		used = total - reclaimable
	}
	return total, used
}

// parseSwapinfo reads the output of `swapinfo -k` (1K blocks).
func parseSwapinfo(out string) (total, used uint64) {
	for _, line := range strings.Split(out, "\n") {
		f := strings.Fields(line)
		if len(f) < 3 {
			continue
		}
		size, err1 := strconv.ParseUint(f[1], 10, 64)
		usedK, err2 := strconv.ParseUint(f[2], 10, 64)
		if err1 != nil || err2 != nil {
			continue // the header line
		}
		if f[0] == "Total" { // several swap devices: the last line sums them up
			return size * 1024, usedK * 1024
		}
		total += size * 1024
		used += usedK * 1024
	}
	return total, used
}

// parseNvidiaSmi reads "name, utilization %, memory used MiB, memory total MiB" lines.
func parseNvidiaSmi(out string) []GPUStat {
	var gpus []GPUStat
	for _, line := range strings.Split(out, "\n") {
		f := strings.Split(line, ",")
		if len(f) < 4 {
			continue
		}
		for i := range f {
			f[i] = strings.TrimSpace(f[i])
		}
		g := GPUStat{Name: f[0], Util: -1}
		if v, err := strconv.ParseFloat(f[1], 64); err == nil {
			g.Util = v
		}
		if v, err := strconv.ParseUint(f[2], 10, 64); err == nil {
			g.MemUsed = v << 20
		}
		if v, err := strconv.ParseUint(f[3], 10, 64); err == nil {
			g.MemTotal = v << 20
		}
		gpus = append(gpus, g)
	}
	return gpus
}

// textBar draws a plain bar (no colours, so it can also be saved in a log).
func textBar(frac float64, width int) string {
	if frac < 0 {
		frac = 0
	}
	if frac > 1 {
		frac = 1
	}
	filled := int(frac*float64(width) + 0.5)
	if filled > width {
		filled = width
	}
	return strings.Repeat("█", filled) + strings.Repeat("░", width-filled)
}

// Lines is the report of :top.
func (s SystemStats) Lines(barWidth int) []string {
	row := func(label string, frac float64, detail string) string {
		return fmt.Sprintf("  %-5s [%s] %3.0f%%  %s", label, textBar(frac, barWidth), frac*100, detail)
	}
	var lines []string
	if s.CPU >= 0 {
		lines = append(lines, row("CPU", s.CPU/100, fmt.Sprintf("%d cores", s.Cores)))
	} else {
		lines = append(lines, "  CPU   not available (or measuring)")
	}
	if s.MemTotal > 0 {
		lines = append(lines, row("RAM", float64(s.MemUsed)/float64(s.MemTotal), humanBytes(int64(s.MemUsed))+" / "+humanBytes(int64(s.MemTotal))))
	} else {
		lines = append(lines, "  RAM   not available")
	}
	if s.SwapTotal > 0 {
		lines = append(lines, row("Swap", float64(s.SwapUsed)/float64(s.SwapTotal), humanBytes(int64(s.SwapUsed))+" / "+humanBytes(int64(s.SwapTotal))))
	} else {
		lines = append(lines, "  Swap  none")
	}
	if len(s.GPUs) == 0 {
		lines = append(lines, "  GPU   not detected (NVIDIA through nvidia-smi, AMD through sysfs)")
	}
	for _, g := range s.GPUs {
		if g.Util >= 0 {
			lines = append(lines, row("GPU", g.Util/100, g.Name))
		} else {
			lines = append(lines, "  GPU   "+g.Name)
		}
		if g.MemTotal > 0 {
			lines = append(lines, row("VRAM", float64(g.MemUsed)/float64(g.MemTotal), humanBytes(int64(g.MemUsed))+" / "+humanBytes(int64(g.MemTotal))))
		}
	}
	if s.Note != "" {
		lines = append(lines, "  "+s.Note)
	}
	return lines
}

// Inline is the short coloured form for the header: CPU 23%  RAM 61%  GPU 40%.
func (s SystemStats) Inline() string {
	var parts []string
	add := func(label string, pct float64) {
		color := colorGreen
		if pct >= 85 {
			color = colorRed
		} else if pct >= 60 {
			color = colorYellow
		}
		parts = append(parts, lipgloss.NewStyle().Foreground(color).Render(fmt.Sprintf("%s %.0f%%", label, pct)))
	}
	if s.CPU >= 0 {
		add("CPU", s.CPU)
	}
	if s.MemTotal > 0 {
		add("RAM", float64(s.MemUsed)/float64(s.MemTotal)*100)
	}
	if s.SwapTotal > 0 && s.SwapUsed > 0 { // swap only matters once it is used
		add("SWAP", float64(s.SwapUsed)/float64(s.SwapTotal)*100)
	}
	for _, g := range s.GPUs {
		if g.Util >= 0 {
			add("GPU", g.Util)
			break
		}
	}
	return strings.Join(parts, "  ")
}
