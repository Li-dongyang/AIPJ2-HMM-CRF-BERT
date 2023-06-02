package main

import (
	"fmt"
	"flags"

	"github.com/VictoriaMetrics/fastcache"
)

weights := fastcache.LoadfromFileOrNew("./ckpt/weights.bin", 128<<20) // 128 MB
config := Config{}

var (
	configFile = flag.String("c", "./config.yaml", "Path to the config file")
)

func main() {
	flag.Parse()
	config.LoadConfig(*configFile)

	if config.Train {
		TrainDataset := Dataset{}
		TrainDataset.LoadDataset(config.TrainFile)
		DevDataset := Dataset{}
		DevDataset.LoadDataset(config.DevFile)

		Model := NewModel(config, weights)
		Model.Train(TrainDataset)

		Model.SaveModel()
		Model.Test(DevDataset)
	} else {
		TestDataset := Dataset{}
		TestDataset.LoadDataset(config.TestFile)

		Model := NewModel(config, weights)
		Model.Test(TestDataset)
	}
}