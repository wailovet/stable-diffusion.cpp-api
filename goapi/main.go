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
var port string
var backendType string = "CUDA"

// lora-model-dir
var loraModelDir string

var currentPath, _ = nuwa.Helper().GetCurrentPath()

var sdCmd *exec.Cmd
var sdStdinPipe io.WriteCloser

func main() {
	flag.StringVar(&backendType, "backend", "CUDA", "backend type")
	flag.StringVar(&model, "model", "", "model name")
	flag.StringVar(&port, "port", "21777", "port")
	flag.Parse()
	nuwa.Config().Port = port
	nuwa.Config().Host = "127.0.0.1"
	log.Println("Starting the service...")

	execName := "sd-cuda-mmq"
	if backendType == "CPU" {
		execName = "sd-flash-attention"
	}

	execPath := filepath.Join(currentPath, execName)
	taesdModel := filepath.Join(currentPath, "taesd.safetensors")
	if model != "" {
		sdCmd = exec.Command(execPath, "--model", model, "--taesd", taesdModel, "--lora-model-dir", currentPath, "-v")
		sdStdinPipe, _ = sdCmd.StdinPipe()
		sdCmd.Stdout = os.Stdout
		sdCmd.Stderr = os.Stderr
		sdCmd.Start()
	}

	nuwa.Http().HandleFunc("/sdapi/v1/reload", func(ctx nuwa.HttpContext) {
		modelPath := ctx.ParamRequired("model_path")
		log.Println("init_model:", modelPath)

		if sdCmd != nil {
			sdStdinPipe.Close()
			sdCmd.Process.Kill()
		}

		sdCmd = exec.Command(execPath, "--model", modelPath, "--taesd", taesdModel, "--lora-model-dir", currentPath, "-v")
		sdStdinPipe, _ = sdCmd.StdinPipe()
		sdCmd.Stdout = os.Stdout
		sdCmd.Stderr = os.Stderr
		sdCmd.Start()
	})

	nuwa.Http().HandleFunc("/sdapi/v1/genimg", func(ctx nuwa.HttpContext) {
		log.Println("genimg:", ctx.BODY)
		body := ctx.BODY
		sdStdinPipe.Write([]byte(body + "\n"))

		ctx.DisplayByData("ok")
	})

	nuwa.Http().Run()
}
