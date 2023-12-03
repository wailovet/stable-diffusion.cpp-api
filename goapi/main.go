package main

import (
	"flag"
	"io"
	"log"
	"os"
	"os/exec"
	"path/filepath"

	"github.com/wailovet/nuwa"
)

var model string
var taesd string
var port string

// lora-model-dir
var loraModelDir string

var currentPath, _ = nuwa.Helper().GetCurrentPath()

var sdCmd *exec.Cmd
var sdStdinPipe io.WriteCloser

func main() {
	flag.StringVar(&model, "model", "", "model name")
	flag.StringVar(&taesd, "taesd", "", "taesd name")
	flag.StringVar(&port, "port", "21777", "port")
	flag.StringVar(&loraModelDir, "lora-model-dir", "", "lora model dir")
	flag.Parse()
	nuwa.Config().Port = port
	nuwa.Config().Host = "127.0.0.1"
	log.Println("Starting the service...")

	exePath := filepath.Join(currentPath, "sd-cuda-mmq")
	sdCmd = exec.Command(exePath, "-m", model, "--taesd", taesd, "--lora-model-dir", loraModelDir, "-v")
	sdStdinPipe, _ = sdCmd.StdinPipe()
	sdCmd.Stdout = os.Stdout
	sdCmd.Stderr = os.Stderr

	nuwa.Http().HandleFunc("/sdapi/v1/txt2img", func(ctx nuwa.HttpContext) {
		log.Println("txt2img:", ctx.BODY)
		body := ctx.BODY
		sdStdinPipe.Write([]byte(body + "\n"))

		ctx.DisplayByData("ok")
	})
	sdCmd.Start()

	nuwa.Http().Run()
}
