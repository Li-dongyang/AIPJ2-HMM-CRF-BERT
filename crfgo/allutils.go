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
		fmt.Println(err)
		return
	}
	sentences := strings.Split(string(dataBytes), "\n\n")
	dataset.Data = make([][]Pair, len(sentences))
	for i := range sentences {
		pairs := strings.Split(sentences[i], "\n")
		dataset.Data[i] = make([]Pair, 0, len(pairs))
		for j := range pairs {
			words := strings.Split(pairs[j], " ")
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

func (dataset *Dataset) GetSentence(idx int) []Pair {
	return dataset.Data[idx]
}

func (dataset *Dataset) Len() int {
	return len(dataset.Data)
}

type Config struct {
	ModelName    string // Name of the model
	WeightsPath  string // Path to the weights file
	TemplatePath string // Path to the template file
	DatasetPath  string // Path to the dataset file
	TrainFile    string // Path to the train file
	DevFile      string // Path to the dev file
	TestFile     string // Path to the test file
	outPutFile   string // Path to the output file

	Train bool // Train the model ?

	Language  string
	Lr        ProbType       // Learning rate
	BatchSize int            // Batch size, implemented by chan for aggretated gradients
	Epoch     int            // Number of epoch
	NumLabels int            // Number of labels
	Label2Idx map[string]int // Label to index, do not use negtive number
	Idx2Label map[int]string // Index to label, do not use negtive number

	maxConcurrency int // Maximum number of concurrent goroutines
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
	// fmt.Printf("config → %+v\n", config)
}
