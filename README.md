# Product Recommendation System

A content-based product recommendation engine built on a Flipkart-style e-commerce product catalog (~20,000 listings). Instead of relying on user ratings or purchase history (mostly missing in this dataset), it recommends products that are textually similar to a given product — based on product name, brand, category, and description.

## How it works

1. **Load & clean** the raw product catalog (`products.csv`) — parse the messy category-tree strings, drop columns that don't help with text similarity (price, images, ratings, etc.), and handle duplicate listings.
2. **Feature engineering** — combine product name, brand, category, and description into a single text field (`all_meta`) per product.
3. **Vectorize** the text using `TfidfVectorizer` (unigrams to trigrams, English stopwords removed, min document frequency of 10).
4. **Compute similarity** between every pair of products using cosine similarity — two versions are built: one from description alone, one from the combined metadata.
5. **Recommend** — given a product, return the top 10 most similar products by similarity score.

## Getting started

### Requirements

```
pandas
numpy
scikit-learn
```

Install with:

```bash
pip install -r requirements.txt
```

### Data

Place the `products.csv` file (Flipkart product dataset) in the project root before running the notebook. [Link to dataset source, if public]

### Running

Open and run `recommendation.ipynb` top to bottom in Jupyter:

```bash
jupyter notebook recommendation.ipynb
```

## Example

```python
product_recommendation(20, des_sim).unique()
```

Returns the 10 products most similar to product at index 20 (`Sicons Conditioning Conditoner Dog Shampoo`) — mostly other pet-care items, confirming description-based similarity captures the right neighborhood.

## Observations & next steps

- Recommendations from description-only similarity (`des_sim`) were noticeably more coherent than those from the combined metadata similarity (`overall_sim`) in testing — likely the category-tree text needs more cleanup, or fields need different weighting rather than a flat concatenation.
- No user interaction data (clicks, purchases, ratings) was available, so this is purely content-based. A natural extension would be a hybrid model if such data becomes available.
- Could also try a distance-based approach on category tags using `pairwise_distances` / `MultiLabelBinarizer`.

## Author

[Jayden Mander Wankhar Sutnga] — MSc Statistics and Data Science, IIT Kanpur
