# Stopwords Source

This project uses a merged Chinese stopword list generated from these open-source sources:

- Repository: goto456/stopwords
- File: hit_stopwords.txt
- Raw URL: https://raw.githubusercontent.com/goto456/stopwords/master/hit_stopwords.txt

- Repository: goto456/stopwords
- File: cn_stopwords.txt
- Raw URL: https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt

- Repository: goto456/stopwords
- File: baidu_stopwords.txt
- Raw URL: https://raw.githubusercontent.com/goto456/stopwords/master/baidu_stopwords.txt

- Repository: stopwords-iso/stopwords-zh
- File: stopwords-zh.txt
- Raw URL: https://raw.githubusercontent.com/stopwords-iso/stopwords-zh/master/stopwords-zh.txt

Runtime file:
- rag/resources/jieba_stop_words_merged.txt

Coverage notes:
- stopwords-iso/stopwords-zh covers terms such as 说 and 中.
- goto456/baidu_stopwords covers terms such as 没有, 现在, and 已经.
- The extractor also filters pure punctuation/symbol tokens in code, so punctuation like ... or … is removed even if not listed line-by-line in the stopword file.
