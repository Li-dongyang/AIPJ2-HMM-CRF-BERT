package main

import (
	"encoding/binary"
	"math"

	"github.com/VictoriaMetrics/fastcache"
)

func (model *LinearCRF) TrainSentence(sentence []Pair) {
	//TODO
}

func (model *LinearCRF) Infer(sentence []Pair) {
	T := len(sentence)
	numLabels := model.Config.NumLabels
	numFeatures := len(model.Templates)
	dp := make([][]float64, T)
	backpointers := make([][]int, T)
	for i := 0; i < T; i++ {
		dp[i] = make([]float64, numLabels)
		backpointers[i] = make([]int, numLabels)
	}
	cache := model.Weights

	// 初始化第一个位置的动态规划值
	for j := 0; j < numLabels; j++ {
		score := 0.0
		for k := 0; k < numFeatures; k++ {
			weight := model.Templates[k].GetFeatureWeightInfer(sentence, 0, STARTTAG, j, cache)
			score += weight
		}
		dp[0][j] = score
	}

	// 递推计算动态规划值和回溯指针
	for t := 1; t < T; t++ {
		for j := 0; j < numLabels; j++ {
			maxScore := math.Inf(-1)
			maxPrevLabel := 0

			for prevLabel := 0; prevLabel < numLabels; prevLabel++ {
				score := dp[t-1][prevLabel]
				for k := 0; k < numFeatures; k++ {
					weight := model.Templates[k].GetFeatureWeightInfer(sentence, t, prevLabel, j, cache)
					score += weight
				}
				if score > maxScore {
					maxScore = score
					maxPrevLabel = prevLabel
				}
			}
			dp[t][j] = maxScore
			backpointers[t][j] = maxPrevLabel
		}
	}

	// 回溯获取最佳标签序列
	bestSequence := make([]int, T)
	bestLabel := 0
	bestScore := math.Inf(-1)
	for j := 0; j < numLabels; j++ {
		score := dp[T-1][j]
		if score > bestScore {
			bestScore = score
			bestLabel = j
		}
	}
	bestSequence[T-1] = bestLabel
	for t := T - 2; t >= 0; t-- {
		bestLabel = backpointers[t+1][bestLabel]
		bestSequence[t] = bestLabel
	}

	// 将推理结果写回 sentence 的 TAG 属性
	for i, label := range bestSequence {
		sentence[i].Tag = label
	}
}

func SetCache(keybytes []byte, value float64, cache *fastcache.Cache) {
	u := math.Float64bits(value)
	bits := make([]byte, 8)
	binary.LittleEndian.PutUint64(bits, u)
	cache.Set(keybytes, bits)
}

func GetCache(keybytes []byte, cache *fastcache.Cache) float64 {
	bits := make([]byte, 8)
	if _, exists := cache.HasGet(bits, keybytes); !exists {
		return 0.0
	}
	u := binary.LittleEndian.Uint64(bits)
	return math.Float64frombits(u)
}
