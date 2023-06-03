package main

import (
	"encoding/binary"

	"github.com/VictoriaMetrics/fastcache"
)

func (model *LinearCRF) TrainSentence(sentence []Pair) {
	predictSequence := make([]int, len(sentence))
	model.Decode(sentence, predictSequence)
	for i := range model.Templates {
		for j := range sentence {
			model.Templates[i].UpdateFeatureWeight(sentence, j, predictSequence, model.Weights, model.Config.Lr)
		}
	}
}

func (model *LinearCRF) Infer(sentence []Pair) {
	bestSequence := make([]int, len(sentence))
	model.Decode(sentence, bestSequence)
	// 将推理结果写回 sentence 的 TAG 属性
	for i, label := range bestSequence {
		sentence[i].Tag = label
	}
}

func (model *LinearCRF) Decode(sentence []Pair, bestSequence []int) {
	T := len(sentence)
	numLabels := model.Config.NumLabels
	numFeatures := len(model.Templates)
	dp := make([][]ProbType, T)
	backpointers := make([][]int, T)
	for i := 0; i < T; i++ {
		dp[i] = make([]ProbType, numLabels)
		backpointers[i] = make([]int, numLabels)
	}
	cache := model.Weights

	// 初始化第一个位置的动态规划值
	for j := 0; j < numLabels; j++ {
		var score ProbType = 0
		for k := 0; k < numFeatures; k++ {
			weight := model.Templates[k].GetFeatureWeightInfer(sentence, 0, STARTTAG, j, cache)
			score += weight
		}
		dp[0][j] = score
	}

	// 递推计算动态规划值和回溯指针
	for t := 1; t < T; t++ {
		for j := 0; j < numLabels; j++ {
			var maxScore ProbType = -1e9
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
	bestLabel := 0
	if bestSequence == nil || len(bestSequence) != T {
		bestSequence = make([]int, T)
	}
	var bestScore ProbType = -1e9
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
}

func SetCache(keybytes []byte, value ProbType, cache *fastcache.Cache) {
	// u := math.Float64bits(value)
	u := uint64(value)
	bits := make([]byte, 8)
	binary.LittleEndian.PutUint64(bits, u)
	cache.Set(keybytes, bits)
}

func GetCache(keybytes []byte, cache *fastcache.Cache) ProbType {
	bits := make([]byte, 8)
	if _, exists := cache.HasGet(bits, keybytes); !exists {
		return ProbType(0)
	}
	u := binary.LittleEndian.Uint64(bits)
	return ProbType(u)
}
