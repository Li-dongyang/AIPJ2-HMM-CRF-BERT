package main

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"sync"

	"github.com/VictoriaMetrics/fastcache"
	"github.com/panjf2000/ants/v2"
)

const (
	START = "<cls>"
	END   = "<eos>"
	STARTTAG = -1
)

type TemplateFunction struct {
	TemplateName string
	IsUnigram bool
	SamlpeIdx []int
}

func (templateFunction *TemplateFunction) GetFeatureWeight(sentence []Pair, idx int, cache *fastcache.Cache) float64 {
	key := make([]string, 0, len(templateFunction.SamlpeIdx) + 3)
	key = append(key, templateFunction.TemplateName)
	for i := range templateFunction.SamlpeIdx {
		if idx + templateFunction.SamlpeIdx[i] < 0 {
			key = append(key, START)
		} else if idx + templateFunction.SamlpeIdx[i] >= len(sentence) {
			key = append(key, END)
		} else {
			key = append(key, sentence[idx + templateFunction.SamlpeIdx[i]].Word)
		}
	}
	if templateFunction.IsUnigram {
		key = append(key, strconv.Itoa(sentence[idx].Tag))
	} else {
		if idx == 0 {
			key = append(key, strconv.Itoa(STARTTAG), strconv.Itoa(sentence[idx].Tag))
		} else {
			key = append(key, strconv.Itoa(sentence[idx - 1].Tag), strconv.Itoa(sentence[idx].Tag))
		}
	}
	keybytes := []byte(strings.Join(key, "/"))
	return GetCache(keybytes, cache)
}

func (templateFunction *TemplateFunction) GetFeatureWeightInfer(sentence []Pair, idx int, prevTag int, thisTag int, cache *fastcache.Cache) float64 {
	key := make([]string, 0, len(templateFunction.SamlpeIdx) + 3)
	key = append(key, templateFunction.TemplateName)
	for i := range templateFunction.SamlpeIdx {
		if idx + templateFunction.SamlpeIdx[i] < 0 {
			key = append(key, START)
		} else if idx + templateFunction.SamlpeIdx[i] >= len(sentence) {
			key = append(key, END)
		} else {
			key = append(key, sentence[idx + templateFunction.SamlpeIdx[i]].Word)
		}
	}
	if templateFunction.IsUnigram {
		key = append(key, strconv.Itoa(thisTag))
	} else {
		key = append(key, strconv.Itoa(prevTag), strconv.Itoa(thisTag))
	}
	keybytes := []byte(strings.Join(key, "/"))
	return GetCache(keybytes, cache)
}

type LinearCRF struct {
	Weights        *fastcache.Cache
	Config         *Config
	Templates      []TemplateFunction
}

func NewModel(config *Config, weights *fastcache.Cache) *LinearCRF {
	model := &LinearCRF{
		Weights: weights,
		Config:  config,
	}
	model.LoadTemplates()
	return model
}

func (model *LinearCRF) Train(dataset Dataset) {
	defer ants.Release()
	wg := sync.WaitGroup{}
	p, err := ants.NewPoolWithFunc(model.Config.BatchSize, func(i any) {
		defer wg.Done()
		sentence := dataset.GetSentence(i.(int))
		model.TrainSentence(sentence)
	}, ants.WithPreAlloc(true), ants.WithNonblocking(true)) // no block when pool is full
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	for epoch := 0; epoch < model.Config.Epoch; epoch++ {
		dataset.Shuffle()
		for i := range dataset.Data { // train each sentence
			wg.Add(1)
			_ = p.Invoke(i)
		}
		wg.Wait()
		fmt.Println("Epoch", epoch, "finished")
	}
}

func (model *LinearCRF) Test(dataset Dataset) {
	wg := sync.WaitGroup{}
	fmt.Println("Testing...")
	for i := range dataset.Data { // train each sentence
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			model.Infer(dataset.Data[i])
		}(i)
	}
	wg.Wait()
	dataset.Store(model.Config.DatasetPath + model.Config.outPutFile)
}

func (model *LinearCRF) LoadTemplates() {
	templateBtyes, err := os.ReadFile(model.Config.TemplatePath)
	if err != nil {
		fmt.Println(err)
		os.Exit(1)
	}
	templateLines := strings.Split(string(templateBtyes), "\n")
	re := regexp.MustCompile(`\[-?\d+`)
	for i := range templateLines {
		templateLines[i] = strings.Trim(templateLines[i], " \n\t\"")
		if len(templateLines[i]) == 0 || templateLines[i][0] == '#' {
			continue
		}
		template := TemplateFunction{}
		templateparse := strings.Split(templateLines[i], ":")
		template.TemplateName = templateparse[0]
		template.IsUnigram = template.TemplateName[0] == 'U'
		matches := re.FindAllString(templateparse[1], -1)
		for _, match := range matches { // remove the first char, which is '['
			if num, err := strconv.Atoi(match[1:]); err != nil {
				fmt.Println(err)
				os.Exit(1)
			} else {
				template.SamlpeIdx = append(template.SamlpeIdx, num)
			}
		}
	}
}

func (model *LinearCRF) SaveModel() {
	if err := model.Weights.SaveToFileConcurrent(weightsFilePath, config.maxConcurrency); err != nil {
		fmt.Println(err)
	} else {
		fmt.Println("Model saved to", weightsFilePath)
	}
}
