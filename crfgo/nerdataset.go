package main

import (
	"io/ioutil"
	"fmt"
	"strings"
)

type TemplateFunction struct {
	TemplateName string
	IsUnigram bool
	SamlpeIdx []int
}

type Dataset struct {
	Data [][]Pair
}

type Pair struct {
	Word string
	Tag string
}

func (dataset *Dataset) LoadDataset(fname string) {
	dataBytes, err := ioutil.ReadFile(fname)
	if err != nil {
		fmt.Println(err)
		return
	}
	sentences := strings.Split(string(dataBytes), "\n\n")
	for i in range(len(sentences)) {
		pairs := strings.Split(sentences[i], "\n")
		for j in range(len(pairs)) {
			words := strings.Split(pairs[j], " ")
			dataset.Data = append(dataset.Data, Pair{words[0], words[1]})
		}
	}
}

type Config struct {
	ModelName string // Name of the model
	WeightsPath string // Path to the weights file
	DatasetPath string // Path to the dataset file
	TrainFile string // Path to the train file
	DevFile string // Path to the dev file
	TestFile string // Path to the test file

	Language string
	Lr float64 // Learning rate
	NumLaBels int // Number of labels
	Label2Idx map[string]int // Label to index
	Idx2Label map[int]string // Index to label

	Train bool // Train the model
}

fun (config *Config) LoadConfig(fname: string) {
	dataBytes, err := os.ReadFile(fname)
	if err != nil {
		fmt.Println(err)
		return
	}
	err = yaml.Unmarshal(dataBytes, config)
	if err != nil {
		fmt.Println("unmarshal error: ", err)
		return
	}
	// fmt.Printf("config → %+v\n", config)
	return
}