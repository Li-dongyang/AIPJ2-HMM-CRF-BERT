package main

import (
	"fmt"
	"math/rand"
	"os"
	"strings"

	"gopkg.in/yaml.v3"
)

type Dataset struct {
	Data      [][]Pair
	Label2Idx map[string]int
	Idx2Label map[int]string
}

type Pair struct {
	Word string
	Tag  int
}

func (dataset *Dataset) LoadDataset(fname string) {
	dataBytes, err := os.ReadFile(fname)
	if err != nil {
		fmt.Println("loaddata", fname, err)
		return
	}
	sentences := strings.Split(string(dataBytes), "\n\n")
	sentences = sentences[:len(sentences)-1]
	dataset.Data = make([][]Pair, len(sentences))
	for i := range sentences {
		pairs := strings.Split(sentences[i], "\n")
		dataset.Data[i] = make([]Pair, 0, len(pairs))
		for j := range pairs {
			words := strings.Split(pairs[j], " ")
			if len(words) != 2 {
				fmt.Println("loaddata split", fname, "error")
				return
			}
			dataset.Data[i] = append(dataset.Data[i], Pair{words[0], dataset.Label2Idx[words[1]]})
		}
	}
}

func (dataset *Dataset) Store(fname string) {
	dataBytes := make([]byte, 0)

	genLine := func(i, j int) []byte {
		tmp := []string{dataset.Data[i][j].Word, " ", dataset.Idx2Label[dataset.Data[i][j].Tag], "\n"}
		return []byte(strings.Join(tmp, ""))
	}

	for i := range dataset.Data {
		for j := range dataset.Data[i] {
			dataBytes = append(dataBytes, genLine(i, j)...)
		}
		dataBytes = append(dataBytes, []byte("\n")...)
	}

	if err := os.WriteFile(fname, dataBytes, 0644); err != nil { // clear the file and write
		fmt.Println(err)
	}
}

func (dataset *Dataset) Shuffle() {
	rand.Shuffle(len(dataset.Data), func(i, j int) {
		dataset.Data[i], dataset.Data[j] = dataset.Data[j], dataset.Data[i]
	})
}

func (dataset *Dataset) Len() int {
	return len(dataset.Data)
}

type Config struct {
	ModelName    string            `yaml:"ModelName"`    // Name of the model
	WeightsPath  string            `yaml:"WeightsPath"`  // Path to the weights file
	TemplatePath string            `yaml:"TemplatePath"` // Path to the template file
	DatasetPath  string            `yaml:"DatasetPath"`  // Path to the dataset file
	TrainFile    string            `yaml:"TrainFile"`    // Path to the train file
	DevFile      string            `yaml:"DevFile"`      // Path to the dev file
	TestFile     string            `yaml:"TestFile"`     // Path to the test file
	OutPutFile   string            `yaml:"OutPutFile"`   // Path to the output file

	Train         bool              `yaml:"Train"`         // Train the model?
	Language      string            `yaml:"Language"`      // Language
	Lr            ProbType          `yaml:"Lr"`            // Learning rate
	BatchSize     int               `yaml:"BatchSize"`     // Batch size, implemented by chan for aggregated gradients
	Epoch         int               `yaml:"Epoch"`         // Number of epoch
	NumLabels     int               `yaml:"NumLabels"`     // Number of labels
	Label2Idx     map[string]int    `yaml:"Label2Idx"`     // Label to index, do not use negative numbers
	Idx2Label     map[int]string    `yaml:"Idx2Label"`     // Index to label, do not use negative numbers
	MaxConcurrency int               `yaml:"MaxConcurrency"` // Maximum number of concurrent goroutines
}


func (config *Config) LoadConfig(fname string) {
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
	fmt.Printf("config is: %+v\n", config)
}
