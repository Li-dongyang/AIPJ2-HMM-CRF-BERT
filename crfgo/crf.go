package main

import (
	"math"
	"fmt"
	"io/ioutil"
	"strings"
	"os"

	"github.com/VictoriaMetrics/fastcache"
)

type ProbFloat float32

type LinearCRF struct {
	Weights *Cache
	Config *Config
	transitionProb [][]ProbFloat
	initialProb []ProbFloat
}