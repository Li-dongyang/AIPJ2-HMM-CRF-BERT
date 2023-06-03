package main

import (
	"fmt"
	"flag"

	"github.com/VictoriaMetrics/fastcache"
)

const (
	weightsFilePath = "../ckpt/weights.bin"
	maxWeightSize = 1 << 30 // 1 GB
)

var (
	configFile = flag.String("c", "./config.yaml", "Path to the config file")
	weights = fastcache.LoadFromFileOrNew(weightsFilePath, maxWeightSize)
	config = Config{}
)

func main() {
	flag.Parse()
	config.LoadConfig(*configFile)

	if config.Train {
		TrainDataset := Dataset{Label2Idx: config.Label2Idx, Idx2Label: config.Idx2Label}
		TrainDataset.LoadDataset(config.DatasetPath + config.TrainFile)
		fmt.Println("Train dataset size:", TrainDataset.Len())
		DevDataset := Dataset{Label2Idx: config.Label2Idx, Idx2Label: config.Idx2Label}
		DevDataset.LoadDataset(config.DatasetPath + config.DevFile)
		fmt.Println("Dev dataset size:", DevDataset.Len())

		Model := NewModel(&config, weights)
		Model.Train(TrainDataset)

		Model.SaveModel()
		Model.Test(DevDataset)
	} else {
		TestDataset := Dataset{Label2Idx: config.Label2Idx, Idx2Label: config.Idx2Label}
		TestDataset.LoadDataset(config.DatasetPath + config.TestFile)

		Model := NewModel(&config, weights)
		Model.Test(TestDataset)
	}
}