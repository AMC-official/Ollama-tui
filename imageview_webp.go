//go:build webp

package main

// WebP pictures can only be decoded inline in kitty-style terminals if a WebP
// decoder is built in. It needs one extra module, so it is optional:
//
//	go get golang.org/x/image/webp
//	go build -tags webp -o ollama-tui .
//
// Without the tag, WebP pictures are drawn by terminals that decode them
// themselves (WezTerm, iTerm2) or opened in the default viewer.

import _ "golang.org/x/image/webp" // registers the decoder for image.Decode
