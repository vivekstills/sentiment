package sentiment

import (
	"math"
	"sort"
	"strings"
	"unicode"
)

// Document represents a labeled text sample.
type Document struct {
	Text  string
	Label string
}

// NaiveBayesClassifier implements a multinomial Naive Bayes model.
type NaiveBayesClassifier struct {
	classDocCounts  map[string]int
	classWordCounts map[string]map[string]int
	classTotalWords map[string]int
	vocabulary      map[string]struct{}
	totalDocs       int
}

// NewNaiveBayesClassifier returns an empty classifier.
func NewNaiveBayesClassifier() *NaiveBayesClassifier {
	return &NaiveBayesClassifier{
		classDocCounts:  make(map[string]int),
		classWordCounts: make(map[string]map[string]int),
		classTotalWords: make(map[string]int),
		vocabulary:      make(map[string]struct{}),
	}
}

// Reset clears all learned statistics.
func (nb *NaiveBayesClassifier) Reset() {
	nb.classDocCounts = make(map[string]int)
	nb.classWordCounts = make(map[string]map[string]int)
	nb.classTotalWords = make(map[string]int)
	nb.vocabulary = make(map[string]struct{})
	nb.totalDocs = 0
}

// Train ingests a labeled document and updates internal counts.
func (nb *NaiveBayesClassifier) Train(text, label string) {
	nb.totalDocs++
	nb.classDocCounts[label]++

	if _, ok := nb.classWordCounts[label]; !ok {
		nb.classWordCounts[label] = make(map[string]int)
	}

	tokens := tokenize(text)
	for _, token := range tokens {
		if token == "" {
			continue
		}
		nb.vocabulary[token] = struct{}{}
		nb.classWordCounts[label][token]++
		nb.classTotalWords[label]++
	}
}

// TrainBatch trains on every document in the slice.
func (nb *NaiveBayesClassifier) TrainBatch(docs []Document) {
	for _, doc := range docs {
		nb.Train(doc.Text, doc.Label)
	}
}

// Predict scores an unseen text and returns the label with the largest posterior probability.
func (nb *NaiveBayesClassifier) Predict(text string) (string, map[string]float64) {
	tokens := tokenize(text)
	scores := make(map[string]float64)
	vocabSize := float64(len(nb.vocabulary))

	bestLabel := ""
	bestScore := math.Inf(-1)

	for class, docCount := range nb.classDocCounts {
		if docCount == 0 {
			continue
		}
		logProb := math.Log(float64(docCount) / float64(nb.totalDocs))
		totalWords := float64(nb.classTotalWords[class])

		for _, token := range tokens {
			if token == "" {
				continue
			}
			wordCount := float64(nb.classWordCounts[class][token])
			logProb += math.Log((wordCount + 1) / (totalWords + vocabSize))
		}

		scores[class] = logProb
		if logProb > bestScore {
			bestScore = logProb
			bestLabel = class
		}
	}

	return bestLabel, normalizeScores(scores, bestScore)
}

func normalizeScores(scores map[string]float64, bestScore float64) map[string]float64 {
	if len(scores) == 0 {
		return map[string]float64{}
	}

	normalized := make(map[string]float64)
	var sum float64
	for class, logProb := range scores {
		value := math.Exp(logProb - bestScore)
		normalized[class] = value
		sum += value
	}

	if sum == 0 {
		return normalized
	}

	for class := range normalized {
		normalized[class] /= sum
	}

	return normalized
}

// Metrics captures evaluation information on a labeled dataset.
type Metrics struct {
	Total     int
	Correct   int
	Confusion map[string]map[string]int
}

// Accuracy returns the accuracy as a floating point value in [0,1].
func (m Metrics) Accuracy() float64 {
	if m.Total == 0 {
		return 0
	}
	return float64(m.Correct) / float64(m.Total)
}

// Evaluate runs the classifier against a labeled dataset and returns metrics.
func Evaluate(nb *NaiveBayesClassifier, docs []Document) Metrics {
	confusion := make(map[string]map[string]int)
	correct := 0

	for _, doc := range docs {
		predicted, _ := nb.Predict(doc.Text)
		if predicted == doc.Label {
			correct++
		}
		if _, ok := confusion[doc.Label]; !ok {
			confusion[doc.Label] = make(map[string]int)
		}
		confusion[doc.Label][predicted]++
	}

	return Metrics{
		Total:     len(docs),
		Correct:   correct,
		Confusion: confusion,
	}
}

func tokenize(text string) []string {
	lower := strings.ToLower(text)
	return strings.FieldsFunc(lower, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
}

// DefaultDataset exposes a small built-in dataset so the binary can run without external files.
func DefaultDataset() []Document {
	docs := make([]Document, len(defaultTrainingData))
	copy(docs, defaultTrainingData)
	return docs
}

// DemoSentences contains short phrases for quick sanity checks.
var DemoSentences = []string{
	"The storyline was engaging and fun",
	"Support ignored my emails for weeks",
	"Delicious food but the service was slow",
	"What an unforgettable and heartwarming play",
}

var defaultTrainingData = []Document{
	{Text: "I love this phone, it's fantastic", Label: "positive"},
	{Text: "The camera is excellent and pictures are great", Label: "positive"},
	{Text: "Absolutely wonderful experience and amazing service", Label: "positive"},
	{Text: "Such a pleasant surprise, highly recommend it", Label: "positive"},
	{Text: "The user interface is clean and easy to use", Label: "positive"},
	{Text: "What a delightful movie, I enjoyed every minute", Label: "positive"},
	{Text: "This book is inspiring and uplifting", Label: "positive"},
	{Text: "Great taste and perfect texture", Label: "positive"},
	{Text: "The trip was fantastic, we had a blast", Label: "positive"},
	{Text: "Beautiful design and very comfortable", Label: "positive"},
	{Text: "I hate how slow this is", Label: "negative"},
	{Text: "The screen cracked within a day", Label: "negative"},
	{Text: "Terrible service and rude employees", Label: "negative"},
	{Text: "I'm disappointed and won't buy again", Label: "negative"},
	{Text: "The instructions are confusing and useless", Label: "negative"},
	{Text: "Worst purchase I've made this year", Label: "negative"},
	{Text: "The food was cold and tasteless", Label: "negative"},
	{Text: "Boring plot with predictable twists", Label: "negative"},
	{Text: "Not worth the price at all", Label: "negative"},
	{Text: "Customer support never replied", Label: "negative"},
}

// Snapshot captures a serializable view of the trained classifier.
type Snapshot struct {
	ClassDocCounts  map[string]int            `json:"class_doc_counts"`
	ClassWordCounts map[string]map[string]int `json:"class_word_counts"`
	ClassTotalWords map[string]int            `json:"class_total_words"`
	Vocabulary      []string                  `json:"vocabulary"`
	TotalDocs       int                       `json:"total_docs"`
}

// Snapshot returns a deep copy of the current classifier state.
func (nb *NaiveBayesClassifier) Snapshot() Snapshot {
	vocab := make([]string, 0, len(nb.vocabulary))
	for token := range nb.vocabulary {
		vocab = append(vocab, token)
	}
	sort.Strings(vocab)

	return Snapshot{
		ClassDocCounts:  copyIntMap(nb.classDocCounts),
		ClassWordCounts: copyNestedMap(nb.classWordCounts),
		ClassTotalWords: copyIntMap(nb.classTotalWords),
		Vocabulary:      vocab,
		TotalDocs:       nb.totalDocs,
	}
}

// LoadSnapshot replaces the classifier state with the contents of the snapshot.
func (nb *NaiveBayesClassifier) LoadSnapshot(snapshot Snapshot) {
	nb.classDocCounts = copyIntMap(snapshot.ClassDocCounts)
	nb.classWordCounts = copyNestedMap(snapshot.ClassWordCounts)
	nb.classTotalWords = copyIntMap(snapshot.ClassTotalWords)
	nb.vocabulary = make(map[string]struct{}, len(snapshot.Vocabulary))
	for _, token := range snapshot.Vocabulary {
		nb.vocabulary[token] = struct{}{}
	}
	nb.totalDocs = snapshot.TotalDocs
}

func copyIntMap(src map[string]int) map[string]int {
	if src == nil {
		return nil
	}
	dst := make(map[string]int, len(src))
	for k, v := range src {
		dst[k] = v
	}
	return dst
}

func copyNestedMap(src map[string]map[string]int) map[string]map[string]int {
	if src == nil {
		return nil
	}
	dst := make(map[string]map[string]int, len(src))
	for k, inner := range src {
		dst[k] = copyIntMap(inner)
	}
	return dst
}
