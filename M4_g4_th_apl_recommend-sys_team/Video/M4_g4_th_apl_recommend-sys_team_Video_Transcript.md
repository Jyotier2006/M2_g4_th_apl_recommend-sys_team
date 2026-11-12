# Video walkthrough transcript

## 1. Predict a rating and its uncertainty

This project asks how to predict a movie rating and how uncertain that prediction is. MovieLens One M contains just over one million ratings. Only about four and a half percent of possible user and movie pairs are observed. Our model predicts the probability of each star rating. We test repeated inference passes and an undirected graph that connects similar movies.

## 2. An ordinal latent trait model

Each user and movie has an eight dimensional latent vector. The coordinates are learned from ratings, so they do not have fixed genre names. Their dot product, plus biases, gives a score. Thresholds convert a noisy score into one of five star ratings. Given the latent traits and biases, ratings are independent. They are related marginally because many ratings share the same traits.

## 3. ADF sees each rating once

Assumed density filtering visits each rating once and approximates its effect with Gaussian messages. One pass of expectation propagation is identical in our implementation. Later passes remove the old contribution, recompute it using the remaining evidence, and damp the update. Repeated passes improve prediction error here, but increase training time. The Gaussian approximation still misses some posterior dependencies.

## 4. Related movies share evidence

The Markov network places an undirected edge between movies with similar rating patterns or genres. The Gaussian prior penalizes differences between connected movie vectors. When a movie has few ratings, a related movie can help stabilize its vector. We fit maximum a posteriori estimates. Setting gamma to zero removes the edge penalty while keeping the same fitting method. That is our graph ablation.

## 5. Compare methods on the same split

For each of five random seeds, all methods share the same training, validation, and test observations. Only training ratings build the graph. Validation chooses settings, and test ratings are used only for evaluation. We also reran the full seed forty two split. All one hundred forty six compared scores and settings matched, with timings excluded.

## 6. The graph gain is small on clean data

Ten expectation propagation passes reduce average root mean squared error from about point eight nine eight to point eight five five. The graph model scores point eight four nine nine. Compared with the same model without the graph, the gain is about six ten thousandths. Four of five within split bootstrap intervals favor the graph; the fifth includes zero. These splits overlap, so we do not treat them as independent replications. Graph MAP has the lowest average log loss, but its calibration error is slightly higher than graph free MAP.

## 7. Smoothing helps most with weak evidence

We kept only one quarter of the training ratings in one experiment. In another, we changed twenty percent of the training stars and kept the test labels clean. The graph advantage grows under both conditions. This is consistent with neighboring movies sharing useful evidence when individual observations are weak. Cold users remain difficult because movie edges do not provide user preferences. Similarity can also be misleading if an edge connects unrelated movies.

## 8. Use the graph where it earns its cost

Microsoft Research Matchbox motivates our approximate inference. PreRec addresses a different problem, cross domain ranking. Our benchmark does not establish performance against either complete system. The main gain from repeated inference is clear. The graph adds modest value on clean ratings and is most useful in the sparse and noisy settings we tested. A next step is to test time ordered splits and account for uncertainty in the graph model. The package contains the code, configurations, results, and reproduction commands.
