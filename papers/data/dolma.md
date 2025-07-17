# LLM Pretraining Data Sources in Dolma

## Web Data - Common Crawl (2.28T tokens)

### Data Processing Pipeline
1. **CCNet Pipeline** (Wenzek et al., 2020)
   - Language identification with FastText
   - Initial paragraph-level deduplication
   - Removes 84.2% of raw Common Crawl data

2. **Quality Filtering**
   - **Gopher Rules** (Rae et al., 2021): Remove pages with excessive repetition, template text
   - **C4 NoPunc**: Remove paragraphs not ending in punctuation
   - Combined approach removes ~38% of CCNet output

3. **Content Filtering**
   - Custom FastText classifiers for hate speech and NSFW content
   - High threshold (τ = 0.4) to balance safety and data retention
   - Removes 5.5-7.3% of content

4. **Deduplication Stages**
   - URL-level: 53.2% of documents removed
   - Document-level: 14.9% of remaining removed
   - Paragraph-level: 18.7% of paragraphs removed

## Data Composition by Domain

### Code - GitHub via The Stack (411B tokens)

### Source
- **The Stack** (Kocetkov et al., 2022)
  - Pre-deduplicated collection of permissively licensed repositories
  - Collected March 2023

### Filtering Approach
- **RedPajama v1 Rules**: Remove repetitive preambles, long lines, numerical content
- **StarCoder Rules**: Filter by repository stars, comment ratios, code-to-text ratios
- Focus on high-quality, well-documented code

## Social Media - Reddit (89B tokens)

### Source
- **Pushshift Reddit Dataset** (Baumgartner et al., 2020)
  - 378M posts from December 2005 to March 2023
  - Includes submissions and comments

### Processing Decisions
- **Thread Linearization**: Treat submissions and comments as independent documents
- **Quality Filtering**:
  - Remove comments < 500 characters
  - Remove submissions < 400 characters
  - Require minimum 3 upvotes
  - Remove deleted/removed content
  - Exclude 26,123 banned/NSFW subreddits

## Academic Papers - Semantic Scholar (70B tokens)

### Source
- **peS2o Dataset** (Soldaini & Lo, 2023)
  - ~40M open-access papers from S2ORC
  - Pre-cleaned and formatted for language modeling

### Key Features
- Already processed for LM training
- Includes abstracts and full text
- Focus on open-access content only

## Books - Project Gutenberg (6B tokens)

### Collection Method
- Scraped April 2023 archive
- English language books only
- Exact-match deduplication by title

### Content
- 70,000+ public domain books
- Classic literature and historical texts
- No copyright concerns

## Reference - Wikipedia & Wikibooks (4.3B tokens)

### Source
- March 2023 Wikimedia dumps
- English and Simple English editions

### Processing
- WikiExtractor for parsing
- Remove short pages (< 25 words)
- No deduplication needed (unique by design)

## Data Mixture Summary

| Source | Tokens | Documents | Purpose |
|--------|---------|-----------|----------|
| Common Crawl | 2.28T (74.5%) | 3.73B | General web text |
| GitHub | 411B (13.4%) | 210M | Code understanding |
| Reddit | 89B (2.9%) | 377M | Conversational/social |
| Semantic Scholar | 70B (2.3%) | 38.8M | Scientific knowledge |
| Project Gutenberg | 6B (0.2%) | 56K | Literary/narrative |
| Wikipedia | 4.3B (0.1%) | 6.2M | Factual reference |
