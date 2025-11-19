package dataset

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"strings"

	"sentimentbayes/sentiment"
)

// LoadCSV reads text,label pairs from a CSV file.
// The first row can optionally be a header containing "text" and "label".
func LoadCSV(path string) ([]sentiment.Document, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	reader := csv.NewReader(file)
	reader.TrimLeadingSpace = true

	var docs []sentiment.Document
	row := 0

	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			return nil, fmt.Errorf("read dataset line %d: %w", row+1, err)
		}
		if len(record) < 2 {
			continue
		}
		if row == 0 && looksLikeHeader(record) {
			row++
			continue
		}

		text := strings.TrimSpace(record[0])
		label := strings.TrimSpace(record[1])
		if text == "" || label == "" {
			row++
			continue
		}
		docs = append(docs, sentiment.Document{
			Text:  text,
			Label: strings.ToLower(label),
		})
		row++
	}

	if len(docs) == 0 {
		return nil, errors.New("dataset is empty")
	}
	return docs, nil
}

// SplitDataset shuffles the dataset and splits it into train/test slices.
func SplitDataset(docs []sentiment.Document, trainRatio float64, seed int64) ([]sentiment.Document, []sentiment.Document) {
	if len(docs) == 0 {
		return nil, nil
	}
	if len(docs) == 1 {
		return append([]sentiment.Document(nil), docs...), nil
	}
	if trainRatio <= 0 || trainRatio >= 1 {
		trainRatio = 0.8
	}

	shuffled := append([]sentiment.Document(nil), docs...)
	rng := rand.New(rand.NewSource(seed))
	rng.Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})

	trainSize := int(math.Round(trainRatio * float64(len(shuffled))))
	if trainSize <= 0 {
		trainSize = 1
	}
	if trainSize >= len(shuffled) {
		trainSize = len(shuffled) - 1
	}

	train := append([]sentiment.Document(nil), shuffled[:trainSize]...)
	test := append([]sentiment.Document(nil), shuffled[trainSize:]...)
	return train, test
}

func looksLikeHeader(record []string) bool {
	if len(record) < 2 {
		return false
	}
	left := strings.ToLower(strings.TrimSpace(record[0]))
	right := strings.ToLower(strings.TrimSpace(record[1]))
	return strings.Contains(left, "text") && strings.Contains(right, "label")
}
