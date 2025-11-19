package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"sort"
	"time"

	"sentimentbayes/dataset"
	"sentimentbayes/sentiment"
)

var (
	datasetPath      = flag.String("dataset", "data/sample.csv", "Path to CSV dataset with text,label columns")
	splitRatio       = flag.Float64("split", 0.8, "Train/test split ratio for evaluation mode")
	randomSeed       = flag.Int64("seed", time.Now().UnixNano(), "Random seed used when shuffling the dataset")
	mode             = flag.String("mode", "demo", "demo|classify|evaluate|serve")
	textInput        = flag.String("text", "", "Text to classify when using classify mode")
	port             = flag.Int("port", 8080, "Port for the HTTP server when using serve mode")
	loadSnapshotPath = flag.String("load-snapshot", "", "Optional path to a JSON snapshot to load before running")
	saveSnapshotPath = flag.String("save-snapshot", "", "Optional path to write the trained model snapshot (demo|classify|serve)")
	continueTraining = flag.Bool("continue-training", false, "Train on the dataset even when -load-snapshot is provided")
)

func main() {
	flag.Parse()

	docs := loadDataset(*datasetPath)
	if len(docs) == 0 {
		log.Fatal("no training data available")
	}

	classifier := sentiment.NewNaiveBayesClassifier()
	snapshotLoaded, err := loadSnapshotFromDisk(classifier, *loadSnapshotPath)
	if err != nil {
		log.Fatal(err)
	}
	shouldTrain := !snapshotLoaded || *continueTraining

	switch *mode {
	case "demo":
		if err := runDemo(classifier, docs, shouldTrain); err != nil {
			log.Fatal(err)
		}
	case "classify":
		if err := runClassifyMode(classifier, docs, *textInput, shouldTrain); err != nil {
			log.Fatal(err)
		}
	case "evaluate":
		if err := runEvaluationMode(classifier, docs, *splitRatio, *randomSeed); err != nil {
			log.Fatal(err)
		}
	case "serve":
		if err := runServerMode(classifier, docs, *port, shouldTrain); err != nil {
			log.Fatal(err)
		}
	default:
		log.Fatalf("unknown mode %q (expected demo|classify|evaluate|serve)", *mode)
	}
}

func loadDataset(path string) []sentiment.Document {
    docs, err := dataset.LoadCSV(path)
    if err == nil {
        return docs
    }
    log.Printf("warning: %v, falling back to built-in dataset", err)
    return sentiment.DefaultDataset()
}

func runDemo(classifier *sentiment.NaiveBayesClassifier, docs []sentiment.Document, train bool) error {
	if train {
		classifier.TrainBatch(docs)
	}
	if err := saveSnapshotIfNeeded(classifier); err != nil {
		return err
	}
	fmt.Println("Sample predictions:")
	for _, sentence := range sentiment.DemoSentences {
		label, probs := classifier.Predict(sentence)
		fmt.Printf("%q -> %s\n", sentence, label)
		printProbabilities(probs)
	}
	return nil
}

func runClassifyMode(classifier *sentiment.NaiveBayesClassifier, docs []sentiment.Document, text string, train bool) error {
	if text == "" {
		return errors.New("-text is required in classify mode")
	}
	if train {
		classifier.TrainBatch(docs)
	}
	if err := saveSnapshotIfNeeded(classifier); err != nil {
		return err
	}
	label, probs := classifier.Predict(text)
	fmt.Printf("Input: %q\n", text)
	fmt.Printf("Predicted sentiment: %s\n", label)
	printProbabilities(probs)
	return nil
}

func runEvaluationMode(classifier *sentiment.NaiveBayesClassifier, docs []sentiment.Document, split float64, seed int64) error {
    train, test := dataset.SplitDataset(docs, split, seed)
    if len(test) == 0 {
        return errors.New("not enough samples to create a test set; provide a larger dataset")
    }
    classifier.Reset()
    classifier.TrainBatch(train)
    metrics := sentiment.Evaluate(classifier, test)

    fmt.Printf("Train set size: %d\n", len(train))
    fmt.Printf("Test set size: %d\n", len(test))
    fmt.Printf("Accuracy: %.2f%% (%d/%d)\n", metrics.Accuracy()*100, metrics.Correct, metrics.Total)
    fmt.Println("Confusion matrix (actual -> predicted counts):")
    printConfusion(metrics.Confusion)
    return nil
}

func runServerMode(classifier *sentiment.NaiveBayesClassifier, docs []sentiment.Document, port int, train bool) error {
	if train {
		classifier.TrainBatch(docs)
	}
	if err := saveSnapshotIfNeeded(classifier); err != nil {
		return err
	}
	srv := &http.Server{
		Addr:    fmt.Sprintf(":%d", port),
		Handler: buildRouter(classifier),
	}
	log.Printf("Serving sentiment API on http://localhost:%d/classify", port)
	return srv.ListenAndServe()
}

func buildRouter(classifier *sentiment.NaiveBayesClassifier) http.Handler {
    mux := http.NewServeMux()
    mux.HandleFunc("/classify", func(w http.ResponseWriter, r *http.Request) {
        if r.Method != http.MethodPost {
            http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
            return
        }
        var req classifyRequest
        if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
            http.Error(w, "invalid JSON body", http.StatusBadRequest)
            return
        }
        if req.Text == "" {
            http.Error(w, "text is required", http.StatusBadRequest)
            return
        }
        label, probs := classifier.Predict(req.Text)
        resp := classifyResponse{Label: label, Probabilities: probs}
        w.Header().Set("Content-Type", "application/json")
        json.NewEncoder(w).Encode(resp)
    })
    return mux
}

func printProbabilities(probs map[string]float64) {
    if len(probs) == 0 {
        fmt.Println("  no class probabilities available")
        return
    }
    classes := make([]string, 0, len(probs))
    for class := range probs {
        classes = append(classes, class)
    }
    sort.Strings(classes)
    for _, class := range classes {
        fmt.Printf("  %s: %.2f\n", class, probs[class])
    }
}

func printConfusion(confusion map[string]map[string]int) {
    actualLabels := make([]string, 0, len(confusion))
    for label := range confusion {
        actualLabels = append(actualLabels, label)
    }
    sort.Strings(actualLabels)
    for _, actual := range actualLabels {
        predicted := confusion[actual]
        predictedLabels := make([]string, 0, len(predicted))
        for label := range predicted {
            predictedLabels = append(predictedLabels, label)
        }
        sort.Strings(predictedLabels)
        fmt.Printf("  %s ->", actual)
        for _, label := range predictedLabels {
            fmt.Printf(" %s:%d", label, predicted[label])
        }
        fmt.Println()
    }
}

type classifyRequest struct {
    Text string `json:"text"`
}

type classifyResponse struct {
    Label         string             `json:"label"`
    Probabilities map[string]float64 `json:"probabilities"`
}

func loadSnapshotFromDisk(classifier *sentiment.NaiveBayesClassifier, path string) (bool, error) {
	if path == "" {
		return false, nil
	}
	data, err := os.ReadFile(path)
	if err != nil {
		return false, fmt.Errorf("load snapshot: %w", err)
	}
	var snapshot sentiment.Snapshot
	if err := json.Unmarshal(data, &snapshot); err != nil {
		return false, fmt.Errorf("decode snapshot: %w", err)
	}
	classifier.LoadSnapshot(snapshot)
	log.Printf("Loaded snapshot from %s", path)
	return true, nil
}

func saveSnapshotIfNeeded(classifier *sentiment.NaiveBayesClassifier) error {
	if *saveSnapshotPath == "" {
		return nil
	}
	snapshot := classifier.Snapshot()
	payload, err := json.MarshalIndent(snapshot, "", "  ")
	if err != nil {
		return fmt.Errorf("encode snapshot: %w", err)
	}
	if err := os.WriteFile(*saveSnapshotPath, payload, 0o644); err != nil {
		return fmt.Errorf("write snapshot: %w", err)
	}
	log.Printf("Snapshot saved to %s", *saveSnapshotPath)
	return nil
}

func init() {
	log.SetOutput(os.Stdout)
}
