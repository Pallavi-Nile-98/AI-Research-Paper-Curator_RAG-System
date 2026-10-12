# Label review checklist

Dataset `0.1.0-draft` — 30 examples, 0 human-reviewed, 30 UNREVIEWED DRAFT -- results computed against unreviewed labels are provisional

Corpus: 124 arXiv papers, predominantly cs.IR, mostly submitted January 2024, indexed as 4,931 chunks. Questions were written against this specific corpus and are not portable to another one.

## How to use this

Work through the questions in any order. For each one there are three things
to decide, and the third is the one that is easy to skip and most worth doing.

1. **Is the question reasonable?** Would someone actually ask it? A question
   nobody would ask measures nothing useful, however well the system answers it.
2. **Are the labelled papers actually relevant?** Read the abstract shown. If a
   paper does not answer the question, strike it.
3. **Is anything missing?** Under *Retrieved but not labelled* are the papers
   the system ranked highly and was marked wrong for. If one of them does answer
   the question, the label is missing, not the retrieval. This is the failure
   mode a drafted dataset is most prone to, and the only way to catch it is to
   look.

Annotate however is convenient — edit this file directly, or write a list. The
fields under each question are a suggestion:

```
VERDICT:   ok | fix | drop
REMOVE:    arxiv ids that should not be labelled relevant
ADD:       arxiv ids that should be
REWORD:    a better version of the question, if it needs one
```

A `drop` verdict is a legitimate outcome. A question that turns out to be
ambiguous or unanswerable is better removed than left in to add noise.

---

## Why this review matters for the headline result

The measured result is that dense retrieval beat hybrid on this dataset. The
known threat to that result is that **the questions were drafted from paper
abstracts by the same process that built the system**, and paraphrase is exactly
what dense retrieval is good at. If the questions are systematically closer in
wording to the abstracts than a real user's question would be, the dataset
favours dense retrieval by construction and the comparison is not meaningful.

Watch for this while reading. A question that reads like a paraphrase of its
paper's abstract is a question worth rewording into something a person would
actually type.

---

## 1. `q001` · exact_terminology

> **What is DREQ?**

**Labelled relevant:**

- **DREQ: Document Re-Ranking Using Entity-based Query Understanding**
  `2401.05939` · cs.IR · [abstract](https://arxiv.org/abs/2401.05939)

  > While entity-oriented neural IR models have advanced significantly, they often
  > overlook a key nuance: the varying degrees of influence individual entities within
  > a document have on its overall relevance. Addressing this gap, we present DREQ, an
  > entity-oriented dense document re-ranking model. Uniquely, we emphasize the query-
  > relevant entities within a document's representation while simultaneously
  > attenuating the ...

**Why these were labelled** (drafted, not verified):

> DREQ is named in the title as an entity-oriented dense document re-ranking model. A
> bare acronym is the clearest case where lexical matching should beat embeddings, since
> an acronym has little semantic content.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(hybrid #2)_ **Graph Neural Networks for Tabular Data Learning: A Survey with Taxonomy and Directions**
  `2401.02143` · cs.LG · [abstract](https://arxiv.org/abs/2401.02143)

  > In this survey, we dive into Tabular Data Learning (TDL) using Graph Neural
  > Networks (GNNs), a domain where deep learning-based approaches have increasingly
  > shown superior performance in both classification and regression tasks compared to
  > traditional methods. The survey highlights a critical gap in deep neural TDL
  > methods: the underrepresentation of latent correlations among data instances and
  > feature values. GNNs, ...

- _(bm25 #2)_ **What Else Would I Like? A User Simulator using Alternatives for Improved Evaluation of Fashion Conversational Recommendation Systems**
  `2401.05783` · cs.IR · [abstract](https://arxiv.org/abs/2401.05783)

  > In Conversational Recommendation Systems (CRS), a user can provide feedback on
  > recommended items at each interaction turn, leading the CRS towards more desirable
  > recommendations. Currently, different types of CRS offer various possibilities for
  > feedback, i.e., natural language feedback, or answering clarifying questions. In
  > most cases, a user simulator is employed for training as well as evaluating the
  > CRS. Such ...

- _(dense #2, hybrid #3, hybrid_rerank #5)_ **UniRQR: A Unified Model for Retrieval Decision, Query, and Response Generation in Internet-Based Knowledge Dialogue Systems**
  `2401.06811` · cs.IR · [abstract](https://arxiv.org/abs/2401.06811)

  > Knowledge-based dialogue systems with internet retrieval have recently attracted
  > considerable attention from researchers. The dialogue systems overcome a major
  > limitation of traditional knowledge dialogue systems, where the timeliness of
  > knowledge cannot be assured, hence providing greater practical application value.
  > Knowledge-based dialogue systems with internet retrieval can be typically
  > segmented into three ...

- _(hybrid_rerank #2)_ **Foundations of Vector Retrieval**
  `2401.09350` · cs.DS · [abstract](https://arxiv.org/abs/2401.09350)

  > Vectors are universal mathematical objects that can represent text, images,
  > speech, or a mix of these data modalities. That happens regardless of whether data
  > is represented by hand-crafted features or learnt embeddings. Collect a large
  > enough quantity of such vectors and the question of retrieval becomes urgently
  > relevant: Finding vectors that are more similar to a query vector. This monograph
  > is concerned with the ...

- _(bm25 #3)_ **Lifelogging As An Extreme Form of Personal Information Management -- What Lessons To Learn**
  `2401.05767` · cs.IR · [abstract](https://arxiv.org/abs/2401.05767)

  > Personal data includes the digital footprints that we leave behind as part of our
  > everyday activities, both online and offline in the real world. It includes data
  > we collect ourselves, such as from wearables, as well as the data collected by
  > others about our online behaviour and activities. Sometimes we are able to use the
  > personal data we ourselves collect, in order to examine some parts of our lives
  > but for the ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 2. `q002` · exact_terminology

> **DocGraphLM documental graph language model**

**Labelled relevant:**

- **DocGraphLM: Documental Graph Language Model for Information Extraction**
  `2401.02823` · cs.CL · [abstract](https://arxiv.org/abs/2401.02823)

  > Advances in Visually Rich Document Understanding (VrDU) have enabled information
  > extraction and question answering over documents with complex layouts. Two tropes
  > of architectures have emerged -- transformer-based models inspired by LLMs, and
  > Graph Neural Networks. In this paper, we introduce DocGraphLM, a novel framework
  > that combines pre-trained language models with graph semantics. To achieve this,
  > we propose 1) ...

**Why these were labelled** (drafted, not verified):

> Exact system name from the title. Tests whether a coined compound token survives the
> stemming analyzer.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #2, hybrid #2, hybrid_rerank #3)_ **Evaluating Large Language Models in Semantic Parsing for Conversational Question Answering over Knowledge Graphs**
  `2401.01711` · cs.CL · [abstract](https://arxiv.org/abs/2401.01711)

  > Conversational question answering systems often rely on semantic parsing to enable
  > interactive information retrieval, which involves the generation of structured
  > database queries from a natural language input. For information-seeking
  > conversations about facts stored within a knowledge graph, dialogue utterances are
  > transformed into graph queries in a process that is called knowledge-based
  > conversational question ...

- _(dense #2, hybrid #3)_ **Mapping Transformer Leveraged Embeddings for Cross-Lingual Document Representation**
  `2401.06583` · cs.CL · [abstract](https://arxiv.org/abs/2401.06583)

  > Recommendation systems, for documents, have become tools to find relevant content
  > on the Web. However, these systems have limitations when it comes to recommending
  > documents in languages different from the query language, which means they might
  > overlook resources in non-native languages. This research focuses on representing
  > documents across languages by using Transformer Leveraged Document Representations
  > (TLDRs) ...

- _(dense #3, hybrid #4, hybrid_rerank #2)_ **LDA-based Term Profiles for Expert Finding in a Political Setting**
  `2401.10617` · cs.IR · [abstract](https://arxiv.org/abs/2401.10617)

  > A common task in many political institutions (i.e. Parliament) is to find
  > politicians who are experts in a particular field. In order to tackle this
  > problem, the first step is to obtain politician profiles which include their
  > interests, and these can be automatically learned from their speeches. As a
  > politician may have various areas of expertise, one alternative is to use a set of
  > subprofiles, each of which covers ...

- _(bm25 #3, hybrid_rerank #4)_ **Unveiling Bias in Fairness Evaluations of Large Language Models: A Critical Literature Review of Music and Movie Recommendation Systems**
  `2401.04057` · cs.IR · [abstract](https://arxiv.org/abs/2401.04057)

  > The rise of generative artificial intelligence, particularly Large Language Models
  > (LLMs), has intensified the imperative to scrutinize fairness alongside accuracy.
  > Recent studies have begun to investigate fairness evaluations for LLMs within
  > domains such as recommendations. Given that personalization is an intrinsic aspect
  > of recommendation systems, its incorporation into fairness assessments is
  > paramount. Yet, the ...

- _(dense #4, hybrid #5)_ **DREQ: Document Re-Ranking Using Entity-based Query Understanding**
  `2401.05939` · cs.IR · [abstract](https://arxiv.org/abs/2401.05939)

  > While entity-oriented neural IR models have advanced significantly, they often
  > overlook a key nuance: the varying degrees of influence individual entities within
  > a document have on its overall relevance. Addressing this gap, we present DREQ, an
  > entity-oriented dense document re-ranking model. Uniquely, we emphasize the query-
  > relevant entities within a document's representation while simultaneously
  > attenuating the ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 3. `q003` · exact_terminology

> **expected transaction value optimization for precise marketing**

**Labelled relevant:**

- **Expected Transaction Value Optimization for Precise Marketing in FinTech Platforms**
  `2401.01525` · cs.IR · [abstract](https://arxiv.org/abs/2401.01525)

  > FinTech platforms facilitated by digital payments are watching growth rapidly,
  > which enable the distribution of mutual funds personalized to individual investors
  > via mobile Apps. As the important intermediation of financial products investment,
  > these platforms distribute thousands of mutual funds obtaining impressions under
  > guaranteed delivery (GD) strategy required by fund companies. Driven by the profit
  > from fund ...

**Why these were labelled** (drafted, not verified):

> Near-verbatim title phrase. Establishes a ceiling: if lexical retrieval cannot find
> this, something is wrong with the index rather than with the method.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(dense #2, hybrid #2, hybrid_rerank #2)_ **Model-Free Approximate Bayesian Learning for Large-Scale Conversion Funnel Optimization**
  `2401.06710` · cs.LG · [abstract](https://arxiv.org/abs/2401.06710)

  > The flexibility of choosing the ad action as a function of the consumer state is
  > critical for modern-day marketing campaigns. We study the problem of identifying
  > the optimal sequential personalized interventions that maximize the adoption
  > probability for a new product. We model consumer behavior by a conversion funnel
  > that captures the state of each consumer (e.g., interaction history with the firm)
  > and allows the ...

- _(dense #3, hybrid #3, hybrid_rerank #5)_ **Variance Reduction in Ratio Metrics for Efficient Online Experiments**
  `2401.04062` · cs.LG · [abstract](https://arxiv.org/abs/2401.04062)

  > Online controlled experiments, such as A/B-tests, are commonly used by modern tech
  > companies to enable continuous system improvements. Despite their paramount
  > importance, A/B-tests are expensive: by their very definition, a percentage of
  > traffic is assigned an inferior system variant. To ensure statistical significance
  > on top-level metrics, online experiments typically run for several weeks. Even
  > then, a ...

- _(hybrid_rerank #3)_ **Ranking Heterogeneous Search Result Pages using the Interactive Probability Ranking Principle**
  `2401.08267` · cs.IR · [abstract](https://arxiv.org/abs/2401.08267)

  > The Probability Ranking Principle (PRP) ranks search results based on their
  > expected utility derived solely from document contents, often overlooking the
  > nuances of presentation and user interaction. However, with the evolution of
  > Search Engine Result Pages (SERPs), now comprising a variety of result cards, the
  > manner in which these results are presented is pivotal in influencing user
  > engagement and satisfaction. ...

- _(hybrid #4)_ **Plug-in Diffusion Model for Sequential Recommendation**
  `2401.02913` · cs.IR · [abstract](https://arxiv.org/abs/2401.02913)

  > Pioneering efforts have verified the effectiveness of the diffusion models in
  > exploring the informative uncertainty for recommendation. Considering the
  > difference between recommendation and image synthesis tasks, existing methods have
  > undertaken tailored refinements to the diffusion and reverse process. However,
  > these approaches typically use the highest-score item in corpus for user interest
  > prediction, leading to ...

- _(hybrid #5, hybrid_rerank #4)_ **Revealing the Hidden Impact of Top-N Metrics on Optimization in Recommender Systems**
  `2401.08444` · cs.IR · [abstract](https://arxiv.org/abs/2401.08444)

  > The hyperparameters of recommender systems for top-n predictions are typically
  > optimized to enhance the predictive performance of algorithms. Thereby, the
  > optimization algorithm, e.g., grid search or random search, searches for the best
  > hyperparameter configuration according to an optimization-target metric, like nDCG
  > or Precision. In contrast, the optimized algorithm, internally optimizes a
  > different loss function ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 4. `q004` · exact_terminology

> **plug-in diffusion model for sequential recommendation**

**Labelled relevant:**

- **Plug-in Diffusion Model for Sequential Recommendation**
  `2401.02913` · cs.IR · [abstract](https://arxiv.org/abs/2401.02913)

  > Pioneering efforts have verified the effectiveness of the diffusion models in
  > exploring the informative uncertainty for recommendation. Considering the
  > difference between recommendation and image synthesis tasks, existing methods have
  > undertaken tailored refinements to the diffusion and reverse process. However,
  > these approaches typically use the highest-score item in corpus for user interest
  > prediction, leading to ...

**Why these were labelled** (drafted, not verified):

> Title phrase. The abstract confirms diffusion models applied to recommendation rather
> than image synthesis.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #2, dense #2, hybrid #2, hybrid_rerank #2)_ **Privacy-Preserving Sequential Recommendation with Collaborative Confusion**
  `2401.04423` · cs.IR · [abstract](https://arxiv.org/abs/2401.04423)

  > Sequential recommendation has attracted a lot of attention from both academia and
  > industry, however the privacy risks associated to gathering and transferring
  > users' personal interaction data are often underestimated or ignored. Existing
  > privacy-preserving studies are mainly applied to traditional collaborative
  > filtering or matrix factorization rather than sequential recommendation. Moreover,
  > these studies are ...

- _(bm25 #3, dense #5, hybrid #5, hybrid_rerank #5)_ **Lightweight Modality Adaptation to Sequential Recommendation via Correlation Supervision**
  `2401.07257` · cs.IR · [abstract](https://arxiv.org/abs/2401.07257)

  > In Sequential Recommenders (SR), encoding and utilizing modalities in an end-to-
  > end manner is costly in terms of modality encoder sizes. Two-stage approaches can
  > mitigate such concerns, but they suffer from poor performance due to modality
  > forgetting, where the sequential objective overshadows modality representation. We
  > propose a lightweight knowledge distillation solution that preserves both merits:
  > retaining ...

- _(dense #3, hybrid #3, hybrid_rerank #3)_ **MCRPL: A Pretrain, Prompt & Fine-tune Paradigm for Non-overlapping Many-to-one Cross-domain Recommendation**
  `2401.08228` · cs.IR · [abstract](https://arxiv.org/abs/2401.08228)

  > Cross-domain Recommendation (CR) is the task that tends to improve the
  > recommendations in the sparse target domain by leveraging the information from
  > other rich domains. Existing methods of cross-domain recommendation mainly focus
  > on overlapping scenarios by assuming users are totally or partially overlapped,
  > which are taken as bridges to connect different domains. However, this assumption
  > does not always hold since ...

- _(hybrid_rerank #4)_ **ChatGPT for Conversational Recommendation: Refining Recommendations by Reprompting with Feedback**
  `2401.03605` · cs.IR · [abstract](https://arxiv.org/abs/2401.03605)

  > Recommendation algorithms have been pivotal in handling the overwhelming volume of
  > online content. However, these algorithms seldom consider direct user input,
  > resulting in superficial interaction between them. Efforts have been made to
  > include the user directly in the recommendation process through conversation, but
  > these systems too have had limited interactivity. Recently, Large Language Models
  > (LLMs) like ...

- _(dense #4, hybrid #4)_ **MultiSlot ReRanker: A Generic Model-based Re-Ranking Framework in Recommendation Systems**
  `2401.06293` · cs.AI · [abstract](https://arxiv.org/abs/2401.06293)

  > In this paper, we propose a generic model-based re-ranking framework, MultiSlot
  > ReRanker, which simultaneously optimizes relevance, diversity, and freshness.
  > Specifically, our Sequential Greedy Algorithm (SGA) is efficient enough (linear
  > time complexity) for large-scale production recommendation engines. It achieved a
  > lift of $+6\%$ to $ +10\%$ offline Area Under the receiver operating
  > characteristic Curve (AUC) ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 5. `q005` · exact_terminology

> **learning to rank with nested feedback**

**Labelled relevant:**

- **Learning-to-Rank with Nested Feedback**
  `2401.04053` · cs.IR · [abstract](https://arxiv.org/abs/2401.04053)

  > Many platforms on the web present ranked lists of content to users, typically
  > optimized for engagement-, satisfaction- or retention- driven metrics. Advances in
  > the Learning-to-Rank (LTR) research literature have enabled rapid growth in this
  > application area. Several popular interfaces now include nested lists, where users
  > can enter a 2nd-level feed via any given 1st-level item. Naturally, this has
  > implications for ...

**Why these were labelled** (drafted, not verified):

> Title phrase. The abstract describes ranked lists on web platforms and nested
> interfaces.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(dense #2, hybrid #3)_ **InRanker: Distilled Rankers for Zero-shot Information Retrieval**
  `2401.06910` · cs.IR · [abstract](https://arxiv.org/abs/2401.06910)

  > Despite multi-billion parameter neural rankers being common components of state-
  > of-the-art information retrieval pipelines, they are rarely used in production due
  > to the enormous amount of compute required for inference. In this work, we propose
  > a new method for distilling large rankers into their smaller versions focusing on
  > out-of-domain effectiveness. We introduce InRanker, a version of monoT5 distilled
  > from ...

- _(bm25 #2, dense #3, hybrid #2, hybrid_rerank #2)_ **A Deep Learning Approach for Selective Relevance Feedback**
  `2401.11198` · cs.IR · [abstract](https://arxiv.org/abs/2401.11198)

  > Pseudo-relevance feedback (PRF) can enhance average retrieval effectiveness over a
  > sufficiently large number of queries. However, PRF often introduces a drift into
  > the original information need, thus hurting the retrieval effectiveness of several
  > queries. While a selective application of PRF can potentially alleviate this
  > issue, previous approaches have largely relied on unsupervised or feature-based
  > learning to ...

- _(hybrid_rerank #3)_ **On the Influence of Reading Sequences on Knowledge Gain during Web Search**
  `2401.05148` · cs.IR · [abstract](https://arxiv.org/abs/2401.05148)

  > Nowadays, learning increasingly involves the usage of search engines and web
  > resources. The related interdisciplinary research field search as learning aims to
  > understand how people learn on the web. Previous work has investigated several
  > feature classes to predict, for instance, the expected knowledge gain during web
  > search. Therein, eye-tracking features have not been extensively studied so far.
  > In this paper, we ...

- _(hybrid #5, hybrid_rerank #4)_ **QoS-Aware Graph Contrastive Learning for Web Service Recommendation**
  `2401.03162` · cs.IR · [abstract](https://arxiv.org/abs/2401.03162)

  > With the rapid growth of cloud services driven by advancements in web service
  > technology, selecting a high-quality service from a wide range of options has
  > become a complex task. This study aims to address the challenges of data sparsity
  > and the cold-start problem in web service recommendation using Quality of Service
  > (QoS). We propose a novel approach called QoS-aware graph contrastive learning
  > (QAGCL) for web ...

- _(dense #5, hybrid #4)_ **Answer Retrieval in Legal Community Question Answering**
  `2401.04852` · cs.IR · [abstract](https://arxiv.org/abs/2401.04852)

  > The task of answer retrieval in the legal domain aims to help users to seek
  > relevant legal advice from massive amounts of professional responses. Two main
  > challenges hinder applying existing answer retrieval approaches in other domains
  > to the legal domain: (1) a huge knowledge gap between lawyers and non-
  > professionals; and (2) a mix of informal and formal content on legal QA websites.
  > To tackle these challenges, we ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 6. `q006` · conceptual

> **How can language models help when there is not enough training data to train a ranker?**

**Labelled relevant:**

- **Team IELAB at TREC Clinical Trial Track 2023: Enhancing Clinical Trial Retrieval with Neural Rankers and Large Language Models**
  `2401.01566` · cs.IR · [abstract](https://arxiv.org/abs/2401.01566)

  > We describe team ielab from CSIRO and The University of Queensland's approach to
  > the 2023 TREC Clinical Trials Track. Our approach was to use neural rankers but to
  > utilise Large Language Models to overcome the issue of lack of training data for
  > such rankers. Specifically, we employ ChatGPT to generate relevant patient
  > descriptions for randomly selected clinical trials from the corpus. This synthetic
  > dataset, ...

**Why these were labelled** (drafted, not verified):

> The abstract states they employ ChatGPT to generate training data precisely because
> neural rankers lack it. The question paraphrases that problem rather than naming the
> method.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #2, dense #1, hybrid #3, hybrid_rerank #3)_ **Translate-Distill: Learning Cross-Language Dense Retrieval by Translation and Distillation**
  `2401.04810` · cs.IR · [abstract](https://arxiv.org/abs/2401.04810)

  > Prior work on English monolingual retrieval has shown that a cross-encoder trained
  > using a large number of relevance judgments for query-document pairs can be used
  > as a teacher to train more efficient, but similarly effective, dual-encoder
  > student models. Applying a similar knowledge distillation approach to training an
  > efficient dual-encoder model for Cross-Language Information Retrieval (CLIR),
  > where queries and ...

- _(bm25 #3, dense #2, hybrid #2, hybrid_rerank #2)_ **InRanker: Distilled Rankers for Zero-shot Information Retrieval**
  `2401.06910` · cs.IR · [abstract](https://arxiv.org/abs/2401.06910)

  > Despite multi-billion parameter neural rankers being common components of state-
  > of-the-art information retrieval pipelines, they are rarely used in production due
  > to the enormous amount of compute required for inference. In this work, we propose
  > a new method for distilling large rankers into their smaller versions focusing on
  > out-of-domain effectiveness. We introduce InRanker, a version of monoT5 distilled
  > from ...

- _(dense #3, hybrid #5)_ **User Embedding Model for Personalized Language Prompting**
  `2401.04858` · cs.CL · [abstract](https://arxiv.org/abs/2401.04858)

  > Modeling long histories plays a pivotal role in enhancing recommendation systems,
  > allowing to capture user's evolving preferences, resulting in more precise and
  > personalized recommendations. In this study we tackle the challenges of modeling
  > long user histories for preference understanding in natural language.
  > Specifically, we introduce a new User Embedding Module (UEM) that efficiently
  > processes user history in ...

- _(dense #4, hybrid_rerank #4)_ **CAPTAIN at COLIEE 2023: Efficient Methods for Legal Information Retrieval and Entailment Tasks**
  `2401.03551` · cs.CL · [abstract](https://arxiv.org/abs/2401.03551)

  > The Competition on Legal Information Extraction/Entailment (COLIEE) is held
  > annually to encourage advancements in the automatic processing of legal texts.
  > Processing legal documents is challenging due to the intricate structure and
  > meaning of legal language. In this paper, we outline our strategies for tackling
  > Task 2, Task 3, and Task 4 in the COLIEE 2023 competition. Our approach involved
  > utilizing appropriate ...

- _(hybrid #4)_ **Data-CUBE: Data Curriculum for Instruction-based Sentence Representation Learning**
  `2401.03563` · cs.CL · [abstract](https://arxiv.org/abs/2401.03563)

  > Recently, multi-task instruction tuning has been applied into sentence
  > representation learning, which endows the capability of generating specific
  > representations with the guidance of task instruction, exhibiting strong
  > generalization ability on new tasks. However, these methods mostly neglect the
  > potential interference problems across different tasks and instances, which may
  > affect the training and convergence of ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 7. `q007` · conceptual

> **What makes it difficult to trust answers produced by a language model?**

**Labelled relevant:**

- **Navigating Uncertainty: Optimizing API Dependency for Hallucination Reduction in Closed-Book Question Answering**
  `2401.01780` · cs.CL · [abstract](https://arxiv.org/abs/2401.01780)

  > While Large Language Models (LLM) are able to accumulate and restore knowledge,
  > they are still prone to hallucination. Especially when faced with factual
  > questions, LLM cannot only rely on knowledge stored in parameters to guarantee
  > truthful and correct answers. Augmenting these models with the ability to search
  > on external information sources, such as the web, is a promising approach to
  > ground knowledge to retrieve ...

- **Physio: An LLM-Based Physiotherapy Advisor**
  `2401.01825` · cs.CL · [abstract](https://arxiv.org/abs/2401.01825)

  > The capabilities of the most recent language models have increased the interest in
  > integrating them into real-world applications. However, the fact that these models
  > generate plausible, yet incorrect text poses a constraint when considering their
  > use in several domains. Healthcare is a prime example of a domain where text-
  > generative trustworthiness is a hard requirement to safeguard patient well-being.
  > In this ...

**Why these were labelled** (drafted, not verified):

> 2401.01780 states LLMs remain prone to hallucination on factual questions. 2401.01825
> states they generate plausible yet incorrect text, which constrains real-world use.
> Both address the question from different angles.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #1)_ **Unveiling Bias in Fairness Evaluations of Large Language Models: A Critical Literature Review of Music and Movie Recommendation Systems**
  `2401.04057` · cs.IR · [abstract](https://arxiv.org/abs/2401.04057)

  > The rise of generative artificial intelligence, particularly Large Language Models
  > (LLMs), has intensified the imperative to scrutinize fairness alongside accuracy.
  > Recent studies have begun to investigate fairness evaluations for LLMs within
  > domains such as recommendations. Given that personalization is an intrinsic aspect
  > of recommendation systems, its incorporation into fairness assessments is
  > paramount. Yet, the ...

- _(dense #1, hybrid #1, hybrid_rerank #2)_ **Adapting Standard Retrieval Benchmarks to Evaluate Generated Answers**
  `2401.04842` · cs.IR · [abstract](https://arxiv.org/abs/2401.04842)

  > Large language models can now directly generate answers to many factual questions
  > without referencing external sources. Unfortunately, relatively little attention
  > has been paid to methods for evaluating the quality and correctness of these
  > answers, for comparing the performance of one model to another, or for comparing
  > one prompt to another. In addition, the quality of generated answers are rarely
  > directly compared ...

- _(bm25 #2)_ **Cross-modal Retrieval for Knowledge-based Visual Question Answering**
  `2401.05736` · cs.CL · [abstract](https://arxiv.org/abs/2401.05736)

  > Knowledge-based Visual Question Answering about Named Entities is a challenging
  > task that requires retrieving information from a multimodal Knowledge Base. Named
  > entities have diverse visual representations and are therefore difficult to
  > recognize. We argue that cross-modal retrieval may help bridge the semantic gap
  > between an entity and its depictions, and is foremost complementary with mono-
  > modal retrieval. We ...

- _(dense #2, hybrid #3)_ **Comparing Traditional and LLM-based Search for Image Geolocation**
  `2401.10184` · cs.IR · [abstract](https://arxiv.org/abs/2401.10184)

  > Web search engines have long served as indispensable tools for information
  > retrieval; user behavior and query formulation strategies have been well studied.
  > The introduction of search engines powered by large language models (LLMs)
  > suggested more conversational search and new types of query strategies. In this
  > paper, we compare traditional and LLM-based search for the task of image
  > geolocation, i.e., determining the ...

- _(dense #3)_ **Evaluating Trustworthiness of Online News Publishers via Article Classification**
  `2401.01781` · cs.IR · [abstract](https://arxiv.org/abs/2401.01781)

  > The proliferation of low-quality online information in today's era has underscored
  > the need for robust and automatic mechanisms to evaluate the trustworthiness of
  > online news publishers. In this paper, we analyse the trustworthiness of online
  > news media outlets by leveraging a dataset of 4033 news stories from 40 different
  > sources. We aim to infer the trustworthiness level of the source based on the
  > classification ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 8. `q008` · conceptual

> **How do online stores suggest the right clothing size to a shopper?**

**Labelled relevant:**

- **Tailor: Size Recommendations for High-End Fashion Marketplaces**
  `2401.01978` · cs.IR · [abstract](https://arxiv.org/abs/2401.01978)

  > In the ever-changing and dynamic realm of high-end fashion marketplaces, providing
  > accurate and personalized size recommendations has become a critical aspect.
  > Meeting customer expectations in this regard is not only crucial for ensuring
  > their satisfaction but also plays a pivotal role in driving customer retention,
  > which is a key metric for the success of any fashion retailer. We propose a novel
  > sequence ...

**Why these were labelled** (drafted, not verified):

> The abstract describes personalized size recommendations for high-end fashion
> marketplaces. The question avoids the words size recommendation.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #2, hybrid #3, hybrid_rerank #5)_ **Lifelogging As An Extreme Form of Personal Information Management -- What Lessons To Learn**
  `2401.05767` · cs.IR · [abstract](https://arxiv.org/abs/2401.05767)

  > Personal data includes the digital footprints that we leave behind as part of our
  > everyday activities, both online and offline in the real world. It includes data
  > we collect ourselves, such as from wearables, as well as the data collected by
  > others about our online behaviour and activities. Sometimes we are able to use the
  > personal data we ourselves collect, in order to examine some parts of our lives
  > but for the ...

- _(dense #2, hybrid_rerank #2)_ **What Else Would I Like? A User Simulator using Alternatives for Improved Evaluation of Fashion Conversational Recommendation Systems**
  `2401.05783` · cs.IR · [abstract](https://arxiv.org/abs/2401.05783)

  > In Conversational Recommendation Systems (CRS), a user can provide feedback on
  > recommended items at each interaction turn, leading the CRS towards more desirable
  > recommendations. Currently, different types of CRS offer various possibilities for
  > feedback, i.e., natural language feedback, or answering clarifying questions. In
  > most cases, a user simulator is employed for training as well as evaluating the
  > CRS. Such ...

- _(hybrid #2, hybrid_rerank #3)_ **Handling Large-scale Cardinality in building recommendation systems**
  `2401.09572` · cs.IR · [abstract](https://arxiv.org/abs/2401.09572)

  > Effective recommendation systems rely on capturing user preferences, often
  > requiring incorporating numerous features such as universally unique identifiers
  > (UUIDs) of entities. However, the exceptionally high cardinality of UUIDs poses a
  > significant challenge in terms of model degradation and increased model size due
  > to sparsity. This paper presents two innovative techniques to address the
  > challenge of high ...

- _(dense #3)_ **Searching, fast and slow, through product catalogs**
  `2401.00737` · cs.IR · [abstract](https://arxiv.org/abs/2401.00737)

  > String matching algorithms in the presence of abbreviations, such as in Stock
  > Keeping Unit (SKU) product catalogs, remains a relatively unexplored topic. In
  > this paper, we present a unified architecture for SKU search that provides both a
  > real-time suggestion system (based on a Trie data structure) as well as a lower
  > latency search system (making use of character level TF-IDF in combination with
  > language model ...

- _(bm25 #3, hybrid #4)_ **Evaluating Trustworthiness of Online News Publishers via Article Classification**
  `2401.01781` · cs.IR · [abstract](https://arxiv.org/abs/2401.01781)

  > The proliferation of low-quality online information in today's era has underscored
  > the need for robust and automatic mechanisms to evaluate the trustworthiness of
  > online news publishers. In this paper, we analyse the trustworthiness of online
  > news media outlets by leveraging a dataset of 4033 news stories from 40 different
  > sources. We aim to infer the trustworthiness level of the source based on the
  > classification ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 9. `q009` · conceptual

> **How can newly released music be surfaced to listeners who have not heard of it?**

**Labelled relevant:**

- **Let's Get It Started: Fostering the Discoverability of New Releases on Deezer**
  `2401.02827` · cs.IR · [abstract](https://arxiv.org/abs/2401.02827)

  > This paper presents our recent initiatives to foster the discoverability of new
  > releases on the music streaming service Deezer. After introducing our search and
  > recommendation features dedicated to new releases, we outline our shift from
  > editorial to personalized release suggestions using cold start embeddings and
  > contextual bandits. Backed by online experiments, we discuss the advantages of
  > this shift in terms of ...

**Why these were labelled** (drafted, not verified):

> The abstract describes fostering discoverability of new releases on Deezer and a shift
> from editorial to personalised approaches.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #1, hybrid #2, hybrid_rerank #2)_ **On the Effect of Data-Augmentation on Local Embedding Properties in the Contrastive Learning of Music Audio Representations**
  `2401.08889` · cs.SD · [abstract](https://arxiv.org/abs/2401.08889)

  > Audio embeddings are crucial tools in understanding large catalogs of music.
  > Typically embeddings are evaluated on the basis of the performance they provide in
  > a wide range of downstream tasks, however few studies have investigated the local
  > properties of the embedding spaces themselves which are important in nearest
  > neighbor algorithms, commonly used in music search and recommendation. In this
  > work we show that ...

- _(bm25 #4, dense #2, hybrid #3, hybrid_rerank #3)_ **Similar but Faster: Manipulation of Tempo in Music Audio Embeddings for Tempo Prediction and Search**
  `2401.08902` · cs.SD · [abstract](https://arxiv.org/abs/2401.08902)

  > Audio embeddings enable large scale comparisons of the similarity of audio files
  > for applications such as search and recommendation. Due to the subjectivity of
  > audio similarity, it can be desirable to design systems that answer not only
  > whether audio is similar, but similar in what way (e.g., wrt. tempo, mood or
  > genre). Previous works have proposed disentangled embedding spaces where subspaces
  > representing specific, ...

- _(bm25 #3, hybrid #4, hybrid_rerank #4)_ **Unveiling Bias in Fairness Evaluations of Large Language Models: A Critical Literature Review of Music and Movie Recommendation Systems**
  `2401.04057` · cs.IR · [abstract](https://arxiv.org/abs/2401.04057)

  > The rise of generative artificial intelligence, particularly Large Language Models
  > (LLMs), has intensified the imperative to scrutinize fairness alongside accuracy.
  > Recent studies have begun to investigate fairness evaluations for LLMs within
  > domains such as recommendations. Given that personalization is an intrinsic aspect
  > of recommendation systems, its incorporation into fairness assessments is
  > paramount. Yet, the ...

- _(dense #3, hybrid #5, hybrid_rerank #5)_ **SARA: A Collection of Sensitivity-Aware Relevance Assessments**
  `2401.05144` · cs.IR · [abstract](https://arxiv.org/abs/2401.05144)

  > Large archival collections, such as email or government documents, must be
  > manually reviewed to identify any sensitive information before the collection can
  > be released publicly. Sensitivity classification has received a lot of attention
  > in the literature. However, more recently, there has been increasing interest in
  > developing sensitivity-aware search engines that can provide users with relevant
  > search results, ...

- _(dense #4)_ **The Impact of Differential Privacy on Recommendation Accuracy and Popularity Bias**
  `2401.03883` · cs.IR · [abstract](https://arxiv.org/abs/2401.03883)

  > Collaborative filtering-based recommender systems leverage vast amounts of
  > behavioral user data, which poses severe privacy risks. Thus, often, random noise
  > is added to the data to ensure Differential Privacy (DP). However, to date, it is
  > not well understood, in which ways this impacts personalized recommendations. In
  > this work, we study how DP impacts recommendation accuracy and popularity bias,
  > when applied to the ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 10. `q010` · conceptual

> **What happens when someone deliberately corrupts the data a recommendation system learns from?**

**Labelled relevant:**

- **Poisoning Attacks against Recommender Systems: A Survey**
  `2401.01527` · cs.IR · [abstract](https://arxiv.org/abs/2401.01527)

  > Modern recommender systems (RS) have seen substantial success, yet they remain
  > vulnerable to malicious activities, notably poisoning attacks. These attacks
  > involve injecting malicious data into the training datasets of RS, thereby
  > compromising their integrity and manipulating recommendation outcomes for gaining
  > illicit profits. This survey paper provides a systematic and up-to-date review of
  > the research landscape ...

**Why these were labelled** (drafted, not verified):

> The abstract defines poisoning attacks as injecting malicious data into recommender
> training sets. The question describes the mechanism without naming it.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(dense #3, hybrid #3, hybrid_rerank #1)_ **The Impact of Differential Privacy on Recommendation Accuracy and Popularity Bias**
  `2401.03883` · cs.IR · [abstract](https://arxiv.org/abs/2401.03883)

  > Collaborative filtering-based recommender systems leverage vast amounts of
  > behavioral user data, which poses severe privacy risks. Thus, often, random noise
  > is added to the data to ensure Differential Privacy (DP). However, to date, it is
  > not well understood, in which ways this impacts personalized recommendations. In
  > this work, we study how DP impacts recommendation accuracy and popularity bias,
  > when applied to the ...

- _(bm25 #2, hybrid #4)_ **LDA-based Term Profiles for Expert Finding in a Political Setting**
  `2401.10617` · cs.IR · [abstract](https://arxiv.org/abs/2401.10617)

  > A common task in many political institutions (i.e. Parliament) is to find
  > politicians who are experts in a particular field. In order to tackle this
  > problem, the first step is to obtain politician profiles which include their
  > interests, and these can be automatically learned from their speeches. As a
  > politician may have various areas of expertise, one alternative is to use a set of
  > subprofiles, each of which covers ...

- _(dense #2, hybrid #2, hybrid_rerank #4)_ **Machine Unlearning for Recommendation Systems: An Insight**
  `2401.10942` · cs.IR · [abstract](https://arxiv.org/abs/2401.10942)

  > This review explores machine unlearning (MUL) in recommendation systems,
  > addressing adaptability, personalization, privacy, and bias challenges. Unlike
  > traditional models, MUL dynamically adjusts system knowledge based on shifts in
  > user preferences and ethical considerations. The paper critically examines MUL's
  > basics, real-world applications, and challenges like algorithmic transparency. It
  > sifts through ...

- _(hybrid_rerank #3)_ **Improving Graph Convolutional Networks with Transformer Layer in social-based items recommendation**
  `2401.06436` · cs.LG · [abstract](https://arxiv.org/abs/2401.06436)

  > In this work, we have proposed an approach for improving the GCN for predicting
  > ratings in social networks. Our model is expanded from the standard model with
  > several layers of transformer architecture. The main focus of the paper is on the
  > encoder architecture for node embedding in the network. Using the embedding layer
  > from the graph-based convolution layer, the attention mechanism could rearrange
  > the feature ...

- _(bm25 #3, hybrid #5)_ **Positive unlabeled learning for building recommender systems in a parliamentary setting**
  `2401.10961` · cs.IR · [abstract](https://arxiv.org/abs/2401.10961)

  > Our goal is to learn about the political interests and preferences of the Members
  > of Parliament by mining their parliamentary activity, in order to develop a
  > recommendation/filtering system that, given a stream of documents to be
  > distributed among them, is able to decide which documents should receive each
  > Member of Parliament. We propose to use positive unlabeled learning to tackle this
  > problem, because we only ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 11. `q011` · conceptual

> **How do you pull structured information out of documents with complicated layouts?**

**Labelled relevant:**

- **DocGraphLM: Documental Graph Language Model for Information Extraction**
  `2401.02823` · cs.CL · [abstract](https://arxiv.org/abs/2401.02823)

  > Advances in Visually Rich Document Understanding (VrDU) have enabled information
  > extraction and question answering over documents with complex layouts. Two tropes
  > of architectures have emerged -- transformer-based models inspired by LLMs, and
  > Graph Neural Networks. In this paper, we introduce DocGraphLM, a novel framework
  > that combines pre-trained language models with graph semantics. To achieve this,
  > we propose 1) ...

**Why these were labelled** (drafted, not verified):

> The abstract describes information extraction and question answering over documents
> with complex layouts, under Visually Rich Document Understanding.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(dense #2, hybrid #3)_ **Enhancing Multilingual Information Retrieval in Mixed Human Resources Environments: A RAG Model Implementation for Multicultural Enterprise**
  `2401.01511` · cs.IR · [abstract](https://arxiv.org/abs/2401.01511)

  > The advent of Large Language Models has revolutionized information retrieval,
  > ushering in a new era of expansive knowledge accessibility. While these models
  > excel in providing open-world knowledge, effectively extracting answers in diverse
  > linguistic environments with varying levels of literacy remains a formidable
  > challenge. Retrieval Augmented Generation (RAG) emerges as a promising solution,
  > bridging the gap ...

- _(bm25 #3, hybrid #2, hybrid_rerank #2)_ **Automatic Construction of Multi-faceted User Profiles using Text Clustering and its Application to Expert Recommendation and Filtering Problems**
  `2401.10634` · cs.IR · [abstract](https://arxiv.org/abs/2401.10634)

  > In the information age we are living in today, not only are we interested in
  > accessing multimedia objects such as documents, videos, etc. but also in searching
  > for professional experts, people or celebrities, possibly for professional needs
  > or just for fun. Information access systems need to be able to extract and exploit
  > various sources of information (usually in text format) about such individuals,
  > and to ...

- _(bm25 #2, hybrid #4)_ **Exploiting Duality in Open Information Extraction with Predicate Prompt**
  `2401.11107` · cs.CL · [abstract](https://arxiv.org/abs/2401.11107)

  > Open information extraction (OpenIE) aims to extract the schema-free triplets in
  > the form of (\emph{subject}, \emph{predicate}, \emph{object}) from a given
  > sentence. Compared with general information extraction (IE), OpenIE poses more
  > challenges for the IE models, {especially when multiple complicated triplets exist
  > in a sentence. To extract these complicated triplets more effectively, in this
  > paper we propose a ...

- _(dense #5, hybrid_rerank #3)_ **Reproducibility Analysis and Enhancements for Multi-Aspect Dense Retriever with Aspect Learning**
  `2401.03648` · cs.IR · [abstract](https://arxiv.org/abs/2401.03648)

  > Multi-aspect dense retrieval aims to incorporate aspect information (e.g., brand
  > and category) into dual encoders to facilitate relevance matching. As an early and
  > representative multi-aspect dense retriever, MADRAL learns several extra aspect
  > embeddings and fuses the explicit aspects with an implicit aspect "OTHER" for
  > final representation. MADRAL was evaluated on proprietary data and its code was
  > not released, ...

- _(dense #3, hybrid #5)_ **A case study of Generative AI in MSX Sales Copilot: Improving seller productivity with a real-time question-answering system for content recommendation**
  `2401.04732` · cs.IR · [abstract](https://arxiv.org/abs/2401.04732)

  > In this paper, we design a real-time question-answering system specifically
  > targeted for helping sellers get relevant material/documentation they can share
  > live with their customers or refer to during a call. Taking the Seismic content
  > repository as a relatively large scale example of a diverse dataset of sales
  > material, we demonstrate how LLM embeddings of sellers' queries can be matched
  > with the relevant content. ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 12. `q012` · conceptual

> **How is multilingual search affected by content that mixes human and machine writing?**

**Labelled relevant:**

- **Enhancing Multilingual Information Retrieval in Mixed Human Resources Environments: A RAG Model Implementation for Multicultural Enterprise**
  `2401.01511` · cs.IR · [abstract](https://arxiv.org/abs/2401.01511)

  > The advent of Large Language Models has revolutionized information retrieval,
  > ushering in a new era of expansive knowledge accessibility. While these models
  > excel in providing open-world knowledge, effectively extracting answers in diverse
  > linguistic environments with varying levels of literacy remains a formidable
  > challenge. Retrieval Augmented Generation (RAG) emerges as a promising solution,
  > bridging the gap ...

**Why these were labelled** (drafted, not verified):

> Title and abstract concern multilingual information retrieval in mixed human and
> machine generated content.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #2, dense #3, hybrid #2, hybrid_rerank #1)_ **Translate-Distill: Learning Cross-Language Dense Retrieval by Translation and Distillation**
  `2401.04810` · cs.IR · [abstract](https://arxiv.org/abs/2401.04810)

  > Prior work on English monolingual retrieval has shown that a cross-encoder trained
  > using a large number of relevance judgments for query-document pairs can be used
  > as a teacher to train more efficient, but similarly effective, dual-encoder
  > student models. Applying a similar knowledge distillation approach to training an
  > efficient dual-encoder model for Cross-Language Information Retrieval (CLIR),
  > where queries and ...

- _(dense #2, hybrid #4, hybrid_rerank #3)_ **Mapping Transformer Leveraged Embeddings for Cross-Lingual Document Representation**
  `2401.06583` · cs.CL · [abstract](https://arxiv.org/abs/2401.06583)

  > Recommendation systems, for documents, have become tools to find relevant content
  > on the Web. However, these systems have limitations when it comes to recommending
  > documents in languages different from the query language, which means they might
  > overlook resources in non-native languages. This research focuses on representing
  > documents across languages by using Transformer Leveraged Document Representations
  > (TLDRs) ...

- _(bm25 #5, dense #5, hybrid #3)_ **Large Language Models vs. Search Engines: Evaluating User Preferences Across Varied Information Retrieval Scenarios**
  `2401.05761` · cs.IR · [abstract](https://arxiv.org/abs/2401.05761)

  > This study embarked on a comprehensive exploration of user preferences between
  > Search Engines and Large Language Models (LLMs) in the context of various
  > information retrieval scenarios. Conducted with a sample size of 100 internet
  > users (N=100) from across the United States, the research delved into 20 distinct
  > use cases ranging from factual searches, such as looking up COVID-19 guidelines,
  > to more subjective tasks, ...

- _(bm25 #3)_ **Ranking Heterogeneous Search Result Pages using the Interactive Probability Ranking Principle**
  `2401.08267` · cs.IR · [abstract](https://arxiv.org/abs/2401.08267)

  > The Probability Ranking Principle (PRP) ranks search results based on their
  > expected utility derived solely from document contents, often overlooking the
  > nuances of presentation and user interaction. However, with the evolution of
  > Search Engine Result Pages (SERPs), now comprising a variety of result cards, the
  > manner in which these results are presented is pivotal in influencing user
  > engagement and satisfaction. ...

- _(dense #4, hybrid #5)_ **Comparing Traditional and LLM-based Search for Image Geolocation**
  `2401.10184` · cs.IR · [abstract](https://arxiv.org/abs/2401.10184)

  > Web search engines have long served as indispensable tools for information
  > retrieval; user behavior and query formulation strategies have been well studied.
  > The introduction of search engines powered by large language models (LLMs)
  > suggested more conversational search and new types of query strategies. In this
  > paper, we compare traditional and LLM-based search for the task of image
  > geolocation, i.e., determining the ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 13. `q013` · method_comparison

> **How do transformer-based and graph-based approaches to document understanding compare?**

**Labelled relevant:**

- **DocGraphLM: Documental Graph Language Model for Information Extraction**
  `2401.02823` · cs.CL · [abstract](https://arxiv.org/abs/2401.02823)

  > Advances in Visually Rich Document Understanding (VrDU) have enabled information
  > extraction and question answering over documents with complex layouts. Two tropes
  > of architectures have emerged -- transformer-based models inspired by LLMs, and
  > Graph Neural Networks. In this paper, we introduce DocGraphLM, a novel framework
  > that combines pre-trained language models with graph semantics. To achieve this,
  > we propose 1) ...

**Why these were labelled** (drafted, not verified):

> The abstract explicitly contrasts two tropes of architectures, transformer-based and
> graph-based. A single-paper comparison, which is why only one paper is labelled.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #2, hybrid #5, hybrid_rerank #4)_ **DREQ: Document Re-Ranking Using Entity-based Query Understanding**
  `2401.05939` · cs.IR · [abstract](https://arxiv.org/abs/2401.05939)

  > While entity-oriented neural IR models have advanced significantly, they often
  > overlook a key nuance: the varying degrees of influence individual entities within
  > a document have on its overall relevance. Addressing this gap, we present DREQ, an
  > entity-oriented dense document re-ranking model. Uniquely, we emphasize the query-
  > relevant entities within a document's representation while simultaneously
  > attenuating the ...

- _(bm25 #5, dense #2, hybrid #2, hybrid_rerank #2)_ **Mapping Transformer Leveraged Embeddings for Cross-Lingual Document Representation**
  `2401.06583` · cs.CL · [abstract](https://arxiv.org/abs/2401.06583)

  > Recommendation systems, for documents, have become tools to find relevant content
  > on the Web. However, these systems have limitations when it comes to recommending
  > documents in languages different from the query language, which means they might
  > overlook resources in non-native languages. This research focuses on representing
  > documents across languages by using Transformer Leveraged Document Representations
  > (TLDRs) ...

- _(bm25 #3, hybrid #4, hybrid_rerank #3)_ **Sparse Meets Dense: A Hybrid Approach to Enhance Scientific Document Retrieval**
  `2401.04055` · cs.IR · [abstract](https://arxiv.org/abs/2401.04055)

  > Traditional information retrieval is based on sparse bag-of-words vector
  > representations of documents and queries. More recent deep-learning approaches
  > have used dense embeddings learned using a transformer-based large language model.
  > We show that on a classic benchmark on scientific document retrieval in the
  > medical domain of cystic fibrosis, that both of these models perform roughly
  > equivalently. Notably, dense ...

- _(bm25 #4, dense #5, hybrid #3)_ **Improving Graph Convolutional Networks with Transformer Layer in social-based items recommendation**
  `2401.06436` · cs.LG · [abstract](https://arxiv.org/abs/2401.06436)

  > In this work, we have proposed an approach for improving the GCN for predicting
  > ratings in social networks. Our model is expanded from the standard model with
  > several layers of transformer architecture. The main focus of the paper is on the
  > encoder architecture for node embedding in the network. Using the embedding layer
  > from the graph-based convolution layer, the attention mechanism could rearrange
  > the feature ...

- _(dense #3)_ **LLMRS: Unlocking Potentials of LLM-Based Recommender Systems for Software Purchase**
  `2401.06676` · cs.IR · [abstract](https://arxiv.org/abs/2401.06676)

  > Recommendation systems are ubiquitous, from Spotify playlist suggestions to Amazon
  > product suggestions. Nevertheless, depending on the methodology or the dataset,
  > these systems typically fail to capture user preferences and generate general
  > recommendations. Recent advancements in Large Language Models (LLM) offer
  > promising results for analyzing user queries. However, employing these models to
  > capture user ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 14. `q014` · method_comparison

> **What are the different ways of evaluating answers a language model generates directly?**

**Labelled relevant:**

- **Adapting Standard Retrieval Benchmarks to Evaluate Generated Answers**
  `2401.04842` · cs.IR · [abstract](https://arxiv.org/abs/2401.04842)

  > Large language models can now directly generate answers to many factual questions
  > without referencing external sources. Unfortunately, relatively little attention
  > has been paid to methods for evaluating the quality and correctness of these
  > answers, for comparing the performance of one model to another, or for comparing
  > one prompt to another. In addition, the quality of generated answers are rarely
  > directly compared ...

- **Evaluating Large Language Models in Semantic Parsing for Conversational Question Answering over Knowledge Graphs**
  `2401.01711` · cs.CL · [abstract](https://arxiv.org/abs/2401.01711)

  > Conversational question answering systems often rely on semantic parsing to enable
  > interactive information retrieval, which involves the generation of structured
  > database queries from a natural language input. For information-seeking
  > conversations about facts stored within a knowledge graph, dialogue utterances are
  > transformed into graph queries in a process that is called knowledge-based
  > conversational question ...

**Why these were labelled** (drafted, not verified):

> 2401.04842 adapts standard retrieval benchmarks to evaluate generated answers.
> 2401.01711 evaluates LLMs on semantic parsing for conversational QA. Both concern
> evaluating generated output, by different means.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #3, dense #2, hybrid #2, hybrid_rerank #3)_ **Navigating Uncertainty: Optimizing API Dependency for Hallucination Reduction in Closed-Book Question Answering**
  `2401.01780` · cs.CL · [abstract](https://arxiv.org/abs/2401.01780)

  > While Large Language Models (LLM) are able to accumulate and restore knowledge,
  > they are still prone to hallucination. Especially when faced with factual
  > questions, LLM cannot only rely on knowledge stored in parameters to guarantee
  > truthful and correct answers. Augmenting these models with the ability to search
  > on external information sources, such as the web, is a promising approach to
  > ground knowledge to retrieve ...

- _(bm25 #4, hybrid_rerank #2)_ **The Chronicles of RAG: The Retriever, the Chunk and the Generator**
  `2401.07883` · cs.LG · [abstract](https://arxiv.org/abs/2401.07883)

  > Retrieval Augmented Generation (RAG) has become one of the most popular paradigms
  > for enabling LLMs to access external data, and also as a mechanism for grounding
  > to mitigate against hallucinations. When implementing RAG you can face several
  > challenges like effective integration of retrieval models, efficient
  > representation learning, data diversity, computational efficiency optimization,
  > evaluation, and quality of ...

- _(dense #3)_ **Physio: An LLM-Based Physiotherapy Advisor**
  `2401.01825` · cs.CL · [abstract](https://arxiv.org/abs/2401.01825)

  > The capabilities of the most recent language models have increased the interest in
  > integrating them into real-world applications. However, the fact that these models
  > generate plausible, yet incorrect text poses a constraint when considering their
  > use in several domains. Healthcare is a prime example of a domain where text-
  > generative trustworthiness is a hard requirement to safeguard patient well-being.
  > In this ...

- _(dense #5, hybrid #3, hybrid_rerank #5)_ **BibSonomy Meets ChatLLMs for Publication Management: From Chat to Publication Management: Organizing your related work using BibSonomy & LLMs**
  `2401.09092` · cs.IR · [abstract](https://arxiv.org/abs/2401.09092)

  > The ever-growing corpus of scientific literature presents significant challenges
  > for researchers with respect to discovery, management, and annotation of relevant
  > publications. Traditional platforms like Semantic Scholar, BibSonomy, and Zotero
  > offer tools for literature management, but largely require manual laborious and
  > error-prone input of tags and metadata. Here, we introduce a novel retrieval
  > augmented ...

- _(dense #4, hybrid #5)_ **Walert: Putting Conversational Search Knowledge into Action by Building and Evaluating a Large Language Model-Powered Chatbot**
  `2401.07216` · cs.IR · [abstract](https://arxiv.org/abs/2401.07216)

  > Creating and deploying customized applications is crucial for operational success
  > and enriching user experiences in the rapidly evolving modern business world. A
  > prominent facet of modern user experiences is the integration of chatbots or voice
  > assistants. The rapid evolution of Large Language Models (LLMs) has provided a
  > powerful tool to build conversational applications. We present Walert, a
  > customized LLM-based ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 15. `q015` · method_comparison

> **Compare re-ranking documents using entities against re-ranking by vector similarity**

**Labelled relevant:**

- **DREQ: Document Re-Ranking Using Entity-based Query Understanding**
  `2401.05939` · cs.IR · [abstract](https://arxiv.org/abs/2401.05939)

  > While entity-oriented neural IR models have advanced significantly, they often
  > overlook a key nuance: the varying degrees of influence individual entities within
  > a document have on its overall relevance. Addressing this gap, we present DREQ, an
  > entity-oriented dense document re-ranking model. Uniquely, we emphasize the query-
  > relevant entities within a document's representation while simultaneously
  > attenuating the ...

- **Concurrent Brainstorming & Hypothesis Satisfying: An Iterative Framework for Enhanced Retrieval-Augmented Generation (R2CBR3H-SR)**
  `2401.01835` · cs.IT · [abstract](https://arxiv.org/abs/2401.01835)

  > Addressing the complexity of comprehensive information retrieval, this study
  > introduces an innovative, iterative retrieval-augmented generation system. Our
  > approach uniquely integrates a vector-space driven re-ranking mechanism with
  > concurrent brainstorming to expedite the retrieval of highly relevant documents,
  > thereby streamlining the generation of potential queries. This sets the stage for
  > our novel hybrid ...

**Why these were labelled** (drafted, not verified):

> 2401.05939 is entity-oriented dense re-ranking; 2401.01835 uses a vector-space driven
> re-ranking mechanism. Neither compares itself to the other, so answering this needs
> both.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(dense #2, hybrid #2)_ **Adapting Standard Retrieval Benchmarks to Evaluate Generated Answers**
  `2401.04842` · cs.IR · [abstract](https://arxiv.org/abs/2401.04842)

  > Large language models can now directly generate answers to many factual questions
  > without referencing external sources. Unfortunately, relatively little attention
  > has been paid to methods for evaluating the quality and correctness of these
  > answers, for comparing the performance of one model to another, or for comparing
  > one prompt to another. In addition, the quality of generated answers are rarely
  > directly compared ...

- _(bm25 #2, hybrid_rerank #2)_ **MultiSlot ReRanker: A Generic Model-based Re-Ranking Framework in Recommendation Systems**
  `2401.06293` · cs.AI · [abstract](https://arxiv.org/abs/2401.06293)

  > In this paper, we propose a generic model-based re-ranking framework, MultiSlot
  > ReRanker, which simultaneously optimizes relevance, diversity, and freshness.
  > Specifically, our Sequential Greedy Algorithm (SGA) is efficient enough (linear
  > time complexity) for large-scale production recommendation engines. It achieved a
  > lift of $+6\%$ to $ +10\%$ offline Area Under the receiver operating
  > characteristic Curve (AUC) ...

- _(dense #3, hybrid #3)_ **Foundations of Vector Retrieval**
  `2401.09350` · cs.DS · [abstract](https://arxiv.org/abs/2401.09350)

  > Vectors are universal mathematical objects that can represent text, images,
  > speech, or a mix of these data modalities. That happens regardless of whether data
  > is represented by hand-crafted features or learnt embeddings. Collect a large
  > enough quantity of such vectors and the question of retrieval becomes urgently
  > relevant: Finding vectors that are more similar to a query vector. This monograph
  > is concerned with the ...

- _(hybrid_rerank #3)_ **On the selection of the correct number of terms for profile construction: theoretical and empirical analysis**
  `2401.10963` · cs.IR · [abstract](https://arxiv.org/abs/2401.10963)

  > In this paper, we examine the problem of building a user profile from a set of
  > documents. This profile will consist of a subset of the most representative terms
  > in the documents that best represent user preferences or interests. Inspired by
  > the discrete concentration theory we have conducted an axiomatic study of seven
  > properties that a selection function should fulfill: the minimum and maximum
  > uncertainty ...

- _(hybrid_rerank #4)_ **Answer Retrieval in Legal Community Question Answering**
  `2401.04852` · cs.IR · [abstract](https://arxiv.org/abs/2401.04852)

  > The task of answer retrieval in the legal domain aims to help users to seek
  > relevant legal advice from massive amounts of professional responses. Two main
  > challenges hinder applying existing answer retrieval approaches in other domains
  > to the legal domain: (1) a huge knowledge gap between lawyers and non-
  > professionals; and (2) a mix of informal and formal content on legal QA websites.
  > To tackle these challenges, we ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 16. `q016` · method_comparison

> **What approaches exist for reducing hallucination in question answering systems?**

**Labelled relevant:**

- **Navigating Uncertainty: Optimizing API Dependency for Hallucination Reduction in Closed-Book Question Answering**
  `2401.01780` · cs.CL · [abstract](https://arxiv.org/abs/2401.01780)

  > While Large Language Models (LLM) are able to accumulate and restore knowledge,
  > they are still prone to hallucination. Especially when faced with factual
  > questions, LLM cannot only rely on knowledge stored in parameters to guarantee
  > truthful and correct answers. Augmenting these models with the ability to search
  > on external information sources, such as the web, is a promising approach to
  > ground knowledge to retrieve ...

- **Physio: An LLM-Based Physiotherapy Advisor**
  `2401.01825` · cs.CL · [abstract](https://arxiv.org/abs/2401.01825)

  > The capabilities of the most recent language models have increased the interest in
  > integrating them into real-world applications. However, the fact that these models
  > generate plausible, yet incorrect text poses a constraint when considering their
  > use in several domains. Healthcare is a prime example of a domain where text-
  > generative trustworthiness is a hard requirement to safeguard patient well-being.
  > In this ...

**Why these were labelled** (drafted, not verified):

> 2401.01780 optimises API dependency to address hallucination. 2401.01825 constrains an
> LLM advisor because of plausible but incorrect output. Different mitigations for the
> same failure.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #5, dense #2, hybrid #3, hybrid_rerank #4)_ **Enhancing Multilingual Information Retrieval in Mixed Human Resources Environments: A RAG Model Implementation for Multicultural Enterprise**
  `2401.01511` · cs.IR · [abstract](https://arxiv.org/abs/2401.01511)

  > The advent of Large Language Models has revolutionized information retrieval,
  > ushering in a new era of expansive knowledge accessibility. While these models
  > excel in providing open-world knowledge, effectively extracting answers in diverse
  > linguistic environments with varying levels of literacy remains a formidable
  > challenge. Retrieval Augmented Generation (RAG) emerges as a promising solution,
  > bridging the gap ...

- _(bm25 #2, hybrid #5)_ **Answer Retrieval in Legal Community Question Answering**
  `2401.04852` · cs.IR · [abstract](https://arxiv.org/abs/2401.04852)

  > The task of answer retrieval in the legal domain aims to help users to seek
  > relevant legal advice from massive amounts of professional responses. Two main
  > challenges hinder applying existing answer retrieval approaches in other domains
  > to the legal domain: (1) a huge knowledge gap between lawyers and non-
  > professionals; and (2) a mix of informal and formal content on legal QA websites.
  > To tackle these challenges, we ...

- _(bm25 #3, dense #3, hybrid #2, hybrid_rerank #5)_ **Cross-modal Retrieval for Knowledge-based Visual Question Answering**
  `2401.05736` · cs.CL · [abstract](https://arxiv.org/abs/2401.05736)

  > Knowledge-based Visual Question Answering about Named Entities is a challenging
  > task that requires retrieving information from a multimodal Knowledge Base. Named
  > entities have diverse visual representations and are therefore difficult to
  > recognize. We argue that cross-modal retrieval may help bridge the semantic gap
  > between an entity and its depictions, and is foremost complementary with mono-
  > modal retrieval. We ...

- _(hybrid_rerank #2)_ **The Chronicles of RAG: The Retriever, the Chunk and the Generator**
  `2401.07883` · cs.LG · [abstract](https://arxiv.org/abs/2401.07883)

  > Retrieval Augmented Generation (RAG) has become one of the most popular paradigms
  > for enabling LLMs to access external data, and also as a mechanism for grounding
  > to mitigate against hallucinations. When implementing RAG you can face several
  > challenges like effective integration of retrieval models, efficient
  > representation learning, data diversity, computational efficiency optimization,
  > evaluation, and quality of ...

- _(dense #4, hybrid_rerank #3)_ **Concurrent Brainstorming & Hypothesis Satisfying: An Iterative Framework for Enhanced Retrieval-Augmented Generation (R2CBR3H-SR)**
  `2401.01835` · cs.IT · [abstract](https://arxiv.org/abs/2401.01835)

  > Addressing the complexity of comprehensive information retrieval, this study
  > introduces an innovative, iterative retrieval-augmented generation system. Our
  > approach uniquely integrates a vector-space driven re-ranking mechanism with
  > concurrent brainstorming to expedite the retrieval of highly relevant documents,
  > thereby streamlining the generation of potential queries. This sets the stage for
  > our novel hybrid ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 17. `q017` · paper_specific

> **What did team IELAB do for the TREC Clinical Trials track in 2023?**

**Labelled relevant:**

- **Team IELAB at TREC Clinical Trial Track 2023: Enhancing Clinical Trial Retrieval with Neural Rankers and Large Language Models**
  `2401.01566` · cs.IR · [abstract](https://arxiv.org/abs/2401.01566)

  > We describe team ielab from CSIRO and The University of Queensland's approach to
  > the 2023 TREC Clinical Trials Track. Our approach was to use neural rankers but to
  > utilise Large Language Models to overcome the issue of lack of training data for
  > such rankers. Specifically, we employ ChatGPT to generate relevant patient
  > descriptions for randomly selected clinical trials from the corpus. This synthetic
  > dataset, ...

**Why these were labelled** (drafted, not verified):

> Names the team, the track and the year, all present in the title and abstract.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #3, dense #4, hybrid #2, hybrid_rerank #5)_ **What Else Would I Like? A User Simulator using Alternatives for Improved Evaluation of Fashion Conversational Recommendation Systems**
  `2401.05783` · cs.IR · [abstract](https://arxiv.org/abs/2401.05783)

  > In Conversational Recommendation Systems (CRS), a user can provide feedback on
  > recommended items at each interaction turn, leading the CRS towards more desirable
  > recommendations. Currently, different types of CRS offer various possibilities for
  > feedback, i.e., natural language feedback, or answering clarifying questions. In
  > most cases, a user simulator is employed for training as well as evaluating the
  > CRS. Such ...

- _(bm25 #5, dense #2, hybrid #4, hybrid_rerank #2)_ **DREQ: Document Re-Ranking Using Entity-based Query Understanding**
  `2401.05939` · cs.IR · [abstract](https://arxiv.org/abs/2401.05939)

  > While entity-oriented neural IR models have advanced significantly, they often
  > overlook a key nuance: the varying degrees of influence individual entities within
  > a document have on its overall relevance. Addressing this gap, we present DREQ, an
  > entity-oriented dense document re-ranking model. Uniquely, we emphasize the query-
  > relevant entities within a document's representation while simultaneously
  > attenuating the ...

- _(bm25 #2, dense #3, hybrid #3)_ **A Reproducibility Study of Goldilocks: Just-Right Tuning of BERT for TAR**
  `2401.08104` · cs.IR · [abstract](https://arxiv.org/abs/2401.08104)

  > Screening documents is a tedious and time-consuming aspect of high-recall
  > retrieval tasks, such as compiling a systematic literature review, where the goal
  > is to identify all relevant documents for a topic. To help streamline this
  > process, many Technology-Assisted Review (TAR) methods leverage active learning
  > techniques to reduce the number of documents requiring review. BERT-based models
  > have shown high ...

- _(hybrid_rerank #3)_ **Translate-Distill: Learning Cross-Language Dense Retrieval by Translation and Distillation**
  `2401.04810` · cs.IR · [abstract](https://arxiv.org/abs/2401.04810)

  > Prior work on English monolingual retrieval has shown that a cross-encoder trained
  > using a large number of relevance judgments for query-document pairs can be used
  > as a teacher to train more efficient, but similarly effective, dual-encoder
  > student models. Applying a similar knowledge distillation approach to training an
  > efficient dual-encoder model for Cross-Language Information Retrieval (CLIR),
  > where queries and ...

- _(hybrid_rerank #4)_ **Adapting Standard Retrieval Benchmarks to Evaluate Generated Answers**
  `2401.04842` · cs.IR · [abstract](https://arxiv.org/abs/2401.04842)

  > Large language models can now directly generate answers to many factual questions
  > without referencing external sources. Unfortunately, relatively little attention
  > has been paid to methods for evaluating the quality and correctness of these
  > answers, for comparing the performance of one model to another, or for comparing
  > one prompt to another. In addition, the quality of generated answers are rarely
  > directly compared ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 18. `q018` · paper_specific

> **What does the Tailor system do for fashion marketplaces?**

**Labelled relevant:**

- **Tailor: Size Recommendations for High-End Fashion Marketplaces**
  `2401.01978` · cs.IR · [abstract](https://arxiv.org/abs/2401.01978)

  > In the ever-changing and dynamic realm of high-end fashion marketplaces, providing
  > accurate and personalized size recommendations has become a critical aspect.
  > Meeting customer expectations in this regard is not only crucial for ensuring
  > their satisfaction but also plays a pivotal role in driving customer retention,
  > which is a key metric for the success of any fashion retailer. We propose a novel
  > sequence ...

**Why these were labelled** (drafted, not verified):

> Named system from the title. Tailor is also an ordinary English word, so this tests
> whether the system name is matched rather than the generic sense.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #2, dense #2, hybrid #2, hybrid_rerank #3)_ **What Else Would I Like? A User Simulator using Alternatives for Improved Evaluation of Fashion Conversational Recommendation Systems**
  `2401.05783` · cs.IR · [abstract](https://arxiv.org/abs/2401.05783)

  > In Conversational Recommendation Systems (CRS), a user can provide feedback on
  > recommended items at each interaction turn, leading the CRS towards more desirable
  > recommendations. Currently, different types of CRS offer various possibilities for
  > feedback, i.e., natural language feedback, or answering clarifying questions. In
  > most cases, a user simulator is employed for training as well as evaluating the
  > CRS. Such ...

- _(dense #4, hybrid #5, hybrid_rerank #2)_ **Lightweight Modality Adaptation to Sequential Recommendation via Correlation Supervision**
  `2401.07257` · cs.IR · [abstract](https://arxiv.org/abs/2401.07257)

  > In Sequential Recommenders (SR), encoding and utilizing modalities in an end-to-
  > end manner is costly in terms of modality encoder sizes. Two-stage approaches can
  > mitigate such concerns, but they suffer from poor performance due to modality
  > forgetting, where the sequential objective overshadows modality representation. We
  > propose a lightweight knowledge distillation solution that preserves both merits:
  > retaining ...

- _(dense #5, hybrid #3)_ **Searching, fast and slow, through product catalogs**
  `2401.00737` · cs.IR · [abstract](https://arxiv.org/abs/2401.00737)

  > String matching algorithms in the presence of abbreviations, such as in Stock
  > Keeping Unit (SKU) product catalogs, remains a relatively unexplored topic. In
  > this paper, we present a unified architecture for SKU search that provides both a
  > real-time suggestion system (based on a Trie data structure) as well as a lower
  > latency search system (making use of character level TF-IDF in combination with
  > language model ...

- _(dense #3, hybrid #4)_ **Curator: Efficient Indexing for Multi-Tenant Vector Databases**
  `2401.07119` · cs.DB · [abstract](https://arxiv.org/abs/2401.07119)

  > Vector databases have emerged as key enablers for bridging intelligent
  > applications with unstructured data, providing generic search and management
  > support for embedding vectors extracted from the raw unstructured data. As
  > multiple data users can share the same database infrastructure, multi-tenancy
  > support for vector databases is increasingly desirable. This hinges on an
  > efficient filtered search operation, i.e., ...

- _(hybrid_rerank #4)_ **Unveiling Bias in Fairness Evaluations of Large Language Models: A Critical Literature Review of Music and Movie Recommendation Systems**
  `2401.04057` · cs.IR · [abstract](https://arxiv.org/abs/2401.04057)

  > The rise of generative artificial intelligence, particularly Large Language Models
  > (LLMs), has intensified the imperative to scrutinize fairness alongside accuracy.
  > Recent studies have begun to investigate fairness evaluations for LLMs within
  > domains such as recommendations. Given that personalization is an intrinsic aspect
  > of recommendation systems, its incorporation into fairness assessments is
  > paramount. Yet, the ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 19. `q019` · paper_specific

> **What is Physio and what is it used for?**

**Labelled relevant:**

- **Physio: An LLM-Based Physiotherapy Advisor**
  `2401.01825` · cs.CL · [abstract](https://arxiv.org/abs/2401.01825)

  > The capabilities of the most recent language models have increased the interest in
  > integrating them into real-world applications. However, the fact that these models
  > generate plausible, yet incorrect text poses a constraint when considering their
  > use in several domains. Healthcare is a prime example of a domain where text-
  > generative trustworthiness is a hard requirement to safeguard patient well-being.
  > In this ...

**Why these were labelled** (drafted, not verified):

> Named system: an LLM-based physiotherapy advisor.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #3, dense #2, hybrid #2, hybrid_rerank #2)_ **Lifelogging As An Extreme Form of Personal Information Management -- What Lessons To Learn**
  `2401.05767` · cs.IR · [abstract](https://arxiv.org/abs/2401.05767)

  > Personal data includes the digital footprints that we leave behind as part of our
  > everyday activities, both online and offline in the real world. It includes data
  > we collect ourselves, such as from wearables, as well as the data collected by
  > others about our online behaviour and activities. Sometimes we are able to use the
  > personal data we ourselves collect, in order to examine some parts of our lives
  > but for the ...

- _(bm25 #2, hybrid #3)_ **What Else Would I Like? A User Simulator using Alternatives for Improved Evaluation of Fashion Conversational Recommendation Systems**
  `2401.05783` · cs.IR · [abstract](https://arxiv.org/abs/2401.05783)

  > In Conversational Recommendation Systems (CRS), a user can provide feedback on
  > recommended items at each interaction turn, leading the CRS towards more desirable
  > recommendations. Currently, different types of CRS offer various possibilities for
  > feedback, i.e., natural language feedback, or answering clarifying questions. In
  > most cases, a user simulator is employed for training as well as evaluating the
  > CRS. Such ...

- _(hybrid_rerank #3)_ **Divergent Characteristics of Biomedical Research across Publication Types: A Quantitative Analysis on the Aging-related Research**
  `2401.04323` · cs.DL · [abstract](https://arxiv.org/abs/2401.04323)

  > This paper investigates differences in characteristics across publication types
  > for aging-related genetic research. We utilized bibliometric data for five model
  > species retrieved from authoritative databases including PubMed. Publications are
  > classified into types according to PubMed. Results indicate substantial divergence
  > across publication types in attention paid to aging-related research, scopes of
  > studied ...

- _(dense #3)_ **An Information Retrieval and Extraction Tool for Covid-19 Related Papers**
  `2401.16430` · cs.IR · [abstract](https://arxiv.org/abs/2401.16430)

  > Background: The COVID-19 pandemic has caused severe impacts on health systems
  > worldwide. Its critical nature and the increased interest of individuals and
  > organizations to develop countermeasures to the problem has led to a surge of new
  > studies in scientific journals. Objetive: We sought to develop a tool that
  > incorporates, in a novel way, aspects of Information Retrieval (IR) and Extraction
  > (IE) applied to the ...

- _(bm25 #5, hybrid_rerank #4)_ **TRACE: A Time-Relational Approximate Cubing Engine for Fast Data Insights**
  `2401.06336` · cs.IR · [abstract](https://arxiv.org/abs/2401.06336)

  > A large class of data questions can be modeled as identifying important slices of
  > data driven by user defined metrics. This paper presents TRACE, a Time-Relational
  > Approximate Cubing Engine that enables interactive analysis on such slices with a
  > low upfront cost - both in space and computation. It does this by materializing
  > the most important parts of the cube over time enabling interactive querying for a
  > large ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 20. `q020` · paper_specific

> **arXiv 2401.05939**

**Labelled relevant:**

- **DREQ: Document Re-Ranking Using Entity-based Query Understanding**
  `2401.05939` · cs.IR · [abstract](https://arxiv.org/abs/2401.05939)

  > While entity-oriented neural IR models have advanced significantly, they often
  > overlook a key nuance: the varying degrees of influence individual entities within
  > a document have on its overall relevance. Addressing this gap, we present DREQ, an
  > entity-oriented dense document re-ranking model. Uniquely, we emphasize the query-
  > relevant entities within a document's representation while simultaneously
  > attenuating the ...

**Why these were labelled** (drafted, not verified):

> A bare identifier. The clearest possible case for lexical retrieval and the clearest
> failure case for embeddings, since an identifier has no semantic neighbourhood.
> Expected to separate the two retrievers sharply.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #1, hybrid #1)_ **Unveiling Bias in Fairness Evaluations of Large Language Models: A Critical Literature Review of Music and Movie Recommendation Systems**
  `2401.04057` · cs.IR · [abstract](https://arxiv.org/abs/2401.04057)

  > The rise of generative artificial intelligence, particularly Large Language Models
  > (LLMs), has intensified the imperative to scrutinize fairness alongside accuracy.
  > Recent studies have begun to investigate fairness evaluations for LLMs within
  > domains such as recommendations. Given that personalization is an intrinsic aspect
  > of recommendation systems, its incorporation into fairness assessments is
  > paramount. Yet, the ...

- _(dense #1, hybrid_rerank #1)_ **Enhancing Image-Text Matching with Adaptive Feature Aggregation**
  `2401.09725` · cs.IR · [abstract](https://arxiv.org/abs/2401.09725)

  > Image-text matching aims to find matched cross-modal pairs accurately. While
  > current methods often rely on projecting cross-modal features into a common
  > embedding space, they frequently suffer from imbalanced feature representations
  > across different modalities, leading to unreliable retrieval results. To address
  > these limitations, we introduce a novel Feature Enhancement Module that adaptively
  > aggregates ...

- _(bm25 #2, hybrid #4)_ **Translate-Distill: Learning Cross-Language Dense Retrieval by Translation and Distillation**
  `2401.04810` · cs.IR · [abstract](https://arxiv.org/abs/2401.04810)

  > Prior work on English monolingual retrieval has shown that a cross-encoder trained
  > using a large number of relevance judgments for query-document pairs can be used
  > as a teacher to train more efficient, but similarly effective, dual-encoder
  > student models. Applying a similar knowledge distillation approach to training an
  > efficient dual-encoder model for Cross-Language Information Retrieval (CLIR),
  > where queries and ...

- _(bm25 #4, dense #4, hybrid #2)_ **A Reproducibility Study of Goldilocks: Just-Right Tuning of BERT for TAR**
  `2401.08104` · cs.IR · [abstract](https://arxiv.org/abs/2401.08104)

  > Screening documents is a tedious and time-consuming aspect of high-recall
  > retrieval tasks, such as compiling a systematic literature review, where the goal
  > is to identify all relevant documents for a topic. To help streamline this
  > process, many Technology-Assisted Review (TAR) methods leverage active learning
  > techniques to reduce the number of documents requiring review. BERT-based models
  > have shown high ...

- _(dense #2, hybrid_rerank #2)_ **Combining topic modelling and citation network analysis to study case law from the European Court on Human Rights on the right to respect for private and family life**
  `2401.16429` · cs.IR · [abstract](https://arxiv.org/abs/2401.16429)

  > As legal case law databases such as HUDOC continue to grow rapidly, it has become
  > essential for legal researchers to find efficient methods to handle such large-
  > scale data sets. Such case law databases usually consist of the textual content of
  > cases together with the citations between them. This paper focuses on case law
  > from the European Court of Human Rights on Article 8 of the European Convention of
  > Human Rights, ...

</details>

Outcome: bm25 **MISSED** · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 21. `q021` · multi_source

> **What are the risks of using AI systems in healthcare?**

**Labelled relevant:**

- **Physio: An LLM-Based Physiotherapy Advisor**
  `2401.01825` · cs.CL · [abstract](https://arxiv.org/abs/2401.01825)

  > The capabilities of the most recent language models have increased the interest in
  > integrating them into real-world applications. However, the fact that these models
  > generate plausible, yet incorrect text poses a constraint when considering their
  > use in several domains. Healthcare is a prime example of a domain where text-
  > generative trustworthiness is a hard requirement to safeguard patient well-being.
  > In this ...

- **Bridging the Skills Gap: Evaluating an AI-Assisted Provider Platform to Support Care Providers with Empathetic Delivery of Protocolized Therapy**
  `2401.03631` · cs.HC · [abstract](https://arxiv.org/abs/2401.03631)

  > Despite the high prevalence and burden of mental health conditions, there is a
  > global shortage of mental health providers. Artificial Intelligence (AI) methods
  > have been proposed as a way to address this shortage, by supporting providers with
  > less extensive training as they deliver care. To this end, we developed the AI-
  > Assisted Provider Platform (A2P2), a text-based virtual therapy interface that
  > includes a ...

**Why these were labelled** (drafted, not verified):

> 2401.01825 concerns an LLM physiotherapy advisor and the constraint of incorrect
> output. 2401.03631 evaluates an AI-assisted platform for mental health provision.
> Neither alone covers healthcare risk broadly.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #1, hybrid #5)_ **What Else Would I Like? A User Simulator using Alternatives for Improved Evaluation of Fashion Conversational Recommendation Systems**
  `2401.05783` · cs.IR · [abstract](https://arxiv.org/abs/2401.05783)

  > In Conversational Recommendation Systems (CRS), a user can provide feedback on
  > recommended items at each interaction turn, leading the CRS towards more desirable
  > recommendations. Currently, different types of CRS offer various possibilities for
  > feedback, i.e., natural language feedback, or answering clarifying questions. In
  > most cases, a user simulator is employed for training as well as evaluating the
  > CRS. Such ...

- _(dense #1, hybrid #3, hybrid_rerank #2)_ **Artificial intelligence to automate the systematic review of scientific literature**
  `2401.10917` · cs.IR · [abstract](https://arxiv.org/abs/2401.10917)

  > Artificial intelligence (AI) has acquired notorious relevance in modern computing
  > as it effectively solves complex tasks traditionally done by humans. AI provides
  > methods to represent and infer knowledge, efficiently manipulate texts and learn
  > from vast amount of data. These characteristics are applicable in many activities
  > that human find laborious or repetitive, as is the case of the analysis of
  > scientific ...

- _(dense #3, hybrid #2, hybrid_rerank #5)_ **Answer Retrieval in Legal Community Question Answering**
  `2401.04852` · cs.IR · [abstract](https://arxiv.org/abs/2401.04852)

  > The task of answer retrieval in the legal domain aims to help users to seek
  > relevant legal advice from massive amounts of professional responses. Two main
  > challenges hinder applying existing answer retrieval approaches in other domains
  > to the legal domain: (1) a huge knowledge gap between lawyers and non-
  > professionals; and (2) a mix of informal and formal content on legal QA websites.
  > To tackle these challenges, we ...

- _(bm25 #3)_ **A systematic review of geospatial location embedding approaches in large language models: A path to spatial AI systems**
  `2401.10279` · cs.IR · [abstract](https://arxiv.org/abs/2401.10279)

  > Geospatial Location Embedding (GLE) helps a Large Language Model (LLM) assimilate
  > and analyze spatial data. GLE emergence in Geospatial Artificial Intelligence
  > (GeoAI) is precipitated by the need for deeper geospatial awareness in our complex
  > contemporary spaces and the success of LLMs in extracting deep meaning in
  > Generative AI. We searched Google Scholar, Science Direct, and arXiv for papers on
  > geospatial location ...

- _(bm25 #5, hybrid #4, hybrid_rerank #3)_ **Machine Unlearning for Recommendation Systems: An Insight**
  `2401.10942` · cs.IR · [abstract](https://arxiv.org/abs/2401.10942)

  > This review explores machine unlearning (MUL) in recommendation systems,
  > addressing adaptability, personalization, privacy, and bias challenges. Unlike
  > traditional models, MUL dynamically adjusts system knowledge based on shifts in
  > user preferences and ethical considerations. The paper critically examines MUL's
  > basics, real-world applications, and challenges like algorithmic transparency. It
  > sifts through ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 22. `q022` · multi_source

> **How is fairness or trustworthiness assessed for automated systems?**

**Labelled relevant:**

- **Unveiling Bias in Fairness Evaluations of Large Language Models: A Critical Literature Review of Music and Movie Recommendation Systems**
  `2401.04057` · cs.IR · [abstract](https://arxiv.org/abs/2401.04057)

  > The rise of generative artificial intelligence, particularly Large Language Models
  > (LLMs), has intensified the imperative to scrutinize fairness alongside accuracy.
  > Recent studies have begun to investigate fairness evaluations for LLMs within
  > domains such as recommendations. Given that personalization is an intrinsic aspect
  > of recommendation systems, its incorporation into fairness assessments is
  > paramount. Yet, the ...

- **Evaluating Trustworthiness of Online News Publishers via Article Classification**
  `2401.01781` · cs.IR · [abstract](https://arxiv.org/abs/2401.01781)

  > The proliferation of low-quality online information in today's era has underscored
  > the need for robust and automatic mechanisms to evaluate the trustworthiness of
  > online news publishers. In this paper, we analyse the trustworthiness of online
  > news media outlets by leveraging a dataset of 4033 news stories from 40 different
  > sources. We aim to infer the trustworthiness level of the source based on the
  > classification ...

**Why these were labelled** (drafted, not verified):

> 2401.04057 examines bias in fairness evaluations of LLMs. 2401.01781 evaluates
> trustworthiness of online news publishers. Two distinct senses of the question.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(dense #3, hybrid #3, hybrid_rerank #4)_ **The Impact of Differential Privacy on Recommendation Accuracy and Popularity Bias**
  `2401.03883` · cs.IR · [abstract](https://arxiv.org/abs/2401.03883)

  > Collaborative filtering-based recommender systems leverage vast amounts of
  > behavioral user data, which poses severe privacy risks. Thus, often, random noise
  > is added to the data to ensure Differential Privacy (DP). However, to date, it is
  > not well understood, in which ways this impacts personalized recommendations. In
  > this work, we study how DP impacts recommendation accuracy and popularity bias,
  > when applied to the ...

- _(hybrid #5, hybrid_rerank #3)_ **Mitigating Position Bias with Regularization for Recommender Systems**
  `2401.16427` · cs.IR · [abstract](https://arxiv.org/abs/2401.16427)

  > Fairness is a popular research topic in recent years. A research topic closely
  > related to fairness is bias and debiasing. Among different types of bias problems,
  > position bias is one of the most widely encountered symptoms. Position bias means
  > that recommended items on top of the recommendation list has a higher likelihood
  > to be clicked than items on bottom of the same list. To mitigate this problem, we
  > propose to ...

- _(dense #4)_ **SARA: A Collection of Sensitivity-Aware Relevance Assessments**
  `2401.05144` · cs.IR · [abstract](https://arxiv.org/abs/2401.05144)

  > Large archival collections, such as email or government documents, must be
  > manually reviewed to identify any sensitive information before the collection can
  > be released publicly. Sensitivity classification has received a lot of attention
  > in the literature. However, more recently, there has been increasing interest in
  > developing sensitivity-aware search engines that can provide users with relevant
  > search results, ...

- _(hybrid #4)_ **Machine Unlearning for Recommendation Systems: An Insight**
  `2401.10942` · cs.IR · [abstract](https://arxiv.org/abs/2401.10942)

  > This review explores machine unlearning (MUL) in recommendation systems,
  > addressing adaptability, personalization, privacy, and bias challenges. Unlike
  > traditional models, MUL dynamically adjusts system knowledge based on shifts in
  > user preferences and ethical considerations. The paper critically examines MUL's
  > basics, real-world applications, and challenges like algorithmic transparency. It
  > sifts through ...

- _(hybrid_rerank #5)_ **Automated Answer Validation using Text Similarity**
  `2401.08688` · cs.CL · [abstract](https://arxiv.org/abs/2401.08688)

  > Automated answer validation can help improve learning outcomes by providing
  > appropriate feedback to learners, and by making question answering systems and
  > online learning solutions more widely available. There have been some works in
  > science question answering which show that information retrieval methods
  > outperform neural methods, especially in the multiple choice version of this
  > problem. We implement Siamese ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 23. `q023` · multi_source

> **What techniques improve retrieval in specialised professional domains such as law or medicine?**

**Labelled relevant:**

- **Answer Retrieval in Legal Community Question Answering**
  `2401.04852` · cs.IR · [abstract](https://arxiv.org/abs/2401.04852)

  > The task of answer retrieval in the legal domain aims to help users to seek
  > relevant legal advice from massive amounts of professional responses. Two main
  > challenges hinder applying existing answer retrieval approaches in other domains
  > to the legal domain: (1) a huge knowledge gap between lawyers and non-
  > professionals; and (2) a mix of informal and formal content on legal QA websites.
  > To tackle these challenges, we ...

- **Team IELAB at TREC Clinical Trial Track 2023: Enhancing Clinical Trial Retrieval with Neural Rankers and Large Language Models**
  `2401.01566` · cs.IR · [abstract](https://arxiv.org/abs/2401.01566)

  > We describe team ielab from CSIRO and The University of Queensland's approach to
  > the 2023 TREC Clinical Trials Track. Our approach was to use neural rankers but to
  > utilise Large Language Models to overcome the issue of lack of training data for
  > such rankers. Specifically, we employ ChatGPT to generate relevant patient
  > descriptions for randomly selected clinical trials from the corpus. This synthetic
  > dataset, ...

**Why these were labelled** (drafted, not verified):

> 2401.04852 addresses legal answer retrieval and the knowledge gap between lawyers and
> laypeople. 2401.01566 addresses clinical trial retrieval. Each covers one domain only.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #1, dense #3, hybrid #3)_ **Combining topic modelling and citation network analysis to study case law from the European Court on Human Rights on the right to respect for private and family life**
  `2401.16429` · cs.IR · [abstract](https://arxiv.org/abs/2401.16429)

  > As legal case law databases such as HUDOC continue to grow rapidly, it has become
  > essential for legal researchers to find efficient methods to handle such large-
  > scale data sets. Such case law databases usually consist of the textual content of
  > cases together with the citations between them. This paper focuses on case law
  > from the European Court of Human Rights on Article 8 of the European Convention of
  > Human Rights, ...

- _(bm25 #4, hybrid #2)_ **CAPTAIN at COLIEE 2023: Efficient Methods for Legal Information Retrieval and Entailment Tasks**
  `2401.03551` · cs.CL · [abstract](https://arxiv.org/abs/2401.03551)

  > The Competition on Legal Information Extraction/Entailment (COLIEE) is held
  > annually to encourage advancements in the automatic processing of legal texts.
  > Processing legal documents is challenging due to the intricate structure and
  > meaning of legal language. In this paper, we outline our strategies for tackling
  > Task 2, Task 3, and Task 4 in the COLIEE 2023 competition. Our approach involved
  > utilizing appropriate ...

- _(hybrid_rerank #2)_ **Sparse Meets Dense: A Hybrid Approach to Enhance Scientific Document Retrieval**
  `2401.04055` · cs.IR · [abstract](https://arxiv.org/abs/2401.04055)

  > Traditional information retrieval is based on sparse bag-of-words vector
  > representations of documents and queries. More recent deep-learning approaches
  > have used dense embeddings learned using a transformer-based large language model.
  > We show that on a classic benchmark on scientific document retrieval in the
  > medical domain of cystic fibrosis, that both of these models perform roughly
  > equivalently. Notably, dense ...

- _(dense #2)_ **Automatic Construction of Multi-faceted User Profiles using Text Clustering and its Application to Expert Recommendation and Filtering Problems**
  `2401.10634` · cs.IR · [abstract](https://arxiv.org/abs/2401.10634)

  > In the information age we are living in today, not only are we interested in
  > accessing multimedia objects such as documents, videos, etc. but also in searching
  > for professional experts, people or celebrities, possibly for professional needs
  > or just for fun. Information access systems need to be able to extract and exploit
  > various sources of information (usually in text format) about such individuals,
  > and to ...

- _(bm25 #3, hybrid_rerank #3)_ **Improved Learned Sparse Retrieval with Corpus-Specific Vocabularies**
  `2401.06703` · cs.IR · [abstract](https://arxiv.org/abs/2401.06703)

  > We explore leveraging corpus-specific vocabularies that improve both efficiency
  > and effectiveness of learned sparse retrieval systems. We find that pre-training
  > the underlying BERT model on the target corpus, specifically targeting different
  > vocabulary sizes incorporated into the document expansion process, improves
  > retrieval quality by up to 12% while in some scenarios decreasing latency by up to
  > 50%. Our ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 24. `q024` · multi_source

> **How are large language models being evaluated as information retrieval systems?**

**Labelled relevant:**

- **Adapting Standard Retrieval Benchmarks to Evaluate Generated Answers**
  `2401.04842` · cs.IR · [abstract](https://arxiv.org/abs/2401.04842)

  > Large language models can now directly generate answers to many factual questions
  > without referencing external sources. Unfortunately, relatively little attention
  > has been paid to methods for evaluating the quality and correctness of these
  > answers, for comparing the performance of one model to another, or for comparing
  > one prompt to another. In addition, the quality of generated answers are rarely
  > directly compared ...

- **Evaluating Large Language Models in Semantic Parsing for Conversational Question Answering over Knowledge Graphs**
  `2401.01711` · cs.CL · [abstract](https://arxiv.org/abs/2401.01711)

  > Conversational question answering systems often rely on semantic parsing to enable
  > interactive information retrieval, which involves the generation of structured
  > database queries from a natural language input. For information-seeking
  > conversations about facts stored within a knowledge graph, dialogue utterances are
  > transformed into graph queries in a process that is called knowledge-based
  > conversational question ...

- **Unveiling Bias in Fairness Evaluations of Large Language Models: A Critical Literature Review of Music and Movie Recommendation Systems**
  `2401.04057` · cs.IR · [abstract](https://arxiv.org/abs/2401.04057)

  > The rise of generative artificial intelligence, particularly Large Language Models
  > (LLMs), has intensified the imperative to scrutinize fairness alongside accuracy.
  > Recent studies have begun to investigate fairness evaluations for LLMs within
  > domains such as recommendations. Given that personalization is an intrinsic aspect
  > of recommendation systems, its incorporation into fairness assessments is
  > paramount. Yet, the ...

**Why these were labelled** (drafted, not verified):

> Three papers on evaluating LLMs from different angles: adapting retrieval benchmarks,
> semantic parsing for conversational QA, and bias in fairness evaluation.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #1, dense #1, hybrid #1, hybrid_rerank #2)_ **Large Language Models vs. Search Engines: Evaluating User Preferences Across Varied Information Retrieval Scenarios**
  `2401.05761` · cs.IR · [abstract](https://arxiv.org/abs/2401.05761)

  > This study embarked on a comprehensive exploration of user preferences between
  > Search Engines and Large Language Models (LLMs) in the context of various
  > information retrieval scenarios. Conducted with a sample size of 100 internet
  > users (N=100) from across the United States, the research delved into 20 distinct
  > use cases ranging from factual searches, such as looking up COVID-19 guidelines,
  > to more subjective tasks, ...

- _(bm25 #2, hybrid #3, hybrid_rerank #4)_ **Walert: Putting Conversational Search Knowledge into Action by Building and Evaluating a Large Language Model-Powered Chatbot**
  `2401.07216` · cs.IR · [abstract](https://arxiv.org/abs/2401.07216)

  > Creating and deploying customized applications is crucial for operational success
  > and enriching user experiences in the rapidly evolving modern business world. A
  > prominent facet of modern user experiences is the integration of chatbots or voice
  > assistants. The rapid evolution of Large Language Models (LLMs) has provided a
  > powerful tool to build conversational applications. We present Walert, a
  > customized LLM-based ...

- _(dense #3, hybrid #4, hybrid_rerank #5)_ **Enhancing Multilingual Information Retrieval in Mixed Human Resources Environments: A RAG Model Implementation for Multicultural Enterprise**
  `2401.01511` · cs.IR · [abstract](https://arxiv.org/abs/2401.01511)

  > The advent of Large Language Models has revolutionized information retrieval,
  > ushering in a new era of expansive knowledge accessibility. While these models
  > excel in providing open-world knowledge, effectively extracting answers in diverse
  > linguistic environments with varying levels of literacy remains a formidable
  > challenge. Retrieval Augmented Generation (RAG) emerges as a promising solution,
  > bridging the gap ...

- _(dense #4)_ **User Embedding Model for Personalized Language Prompting**
  `2401.04858` · cs.CL · [abstract](https://arxiv.org/abs/2401.04858)

  > Modeling long histories plays a pivotal role in enhancing recommendation systems,
  > allowing to capture user's evolving preferences, resulting in more precise and
  > personalized recommendations. In this study we tackle the challenges of modeling
  > long user histories for preference understanding in natural language.
  > Specifically, we introduce a new User Embedding Module (UEM) that efficiently
  > processes user history in ...

- _(dense #5)_ **Improved Learned Sparse Retrieval with Corpus-Specific Vocabularies**
  `2401.06703` · cs.IR · [abstract](https://arxiv.org/abs/2401.06703)

  > We explore leveraging corpus-specific vocabularies that improve both efficiency
  > and effectiveness of learned sparse retrieval systems. We find that pre-training
  > the underlying BERT model on the target corpus, specifically targeting different
  > vocabulary sizes incorporated into the document expansion process, improves
  > retrieval quality by up to 12% while in some scenarios decreasing latency by up to
  > 50%. Our ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 25. `q025` · multi_source

> **In what ways can recommender systems be manipulated or produce unfair outcomes?**

**Labelled relevant:**

- **Poisoning Attacks against Recommender Systems: A Survey**
  `2401.01527` · cs.IR · [abstract](https://arxiv.org/abs/2401.01527)

  > Modern recommender systems (RS) have seen substantial success, yet they remain
  > vulnerable to malicious activities, notably poisoning attacks. These attacks
  > involve injecting malicious data into the training datasets of RS, thereby
  > compromising their integrity and manipulating recommendation outcomes for gaining
  > illicit profits. This survey paper provides a systematic and up-to-date review of
  > the research landscape ...

- **Unveiling Bias in Fairness Evaluations of Large Language Models: A Critical Literature Review of Music and Movie Recommendation Systems**
  `2401.04057` · cs.IR · [abstract](https://arxiv.org/abs/2401.04057)

  > The rise of generative artificial intelligence, particularly Large Language Models
  > (LLMs), has intensified the imperative to scrutinize fairness alongside accuracy.
  > Recent studies have begun to investigate fairness evaluations for LLMs within
  > domains such as recommendations. Given that personalization is an intrinsic aspect
  > of recommendation systems, its incorporation into fairness assessments is
  > paramount. Yet, the ...

**Why these were labelled** (drafted, not verified):

> 2401.01527 surveys poisoning attacks; 2401.04057 concerns bias in fairness evaluation.
> Manipulation and unfairness are distinct failures and no single paper covers both.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #3, dense #3, hybrid #1, hybrid_rerank #2)_ **The Impact of Differential Privacy on Recommendation Accuracy and Popularity Bias**
  `2401.03883` · cs.IR · [abstract](https://arxiv.org/abs/2401.03883)

  > Collaborative filtering-based recommender systems leverage vast amounts of
  > behavioral user data, which poses severe privacy risks. Thus, often, random noise
  > is added to the data to ensure Differential Privacy (DP). However, to date, it is
  > not well understood, in which ways this impacts personalized recommendations. In
  > this work, we study how DP impacts recommendation accuracy and popularity bias,
  > when applied to the ...

- _(dense #2, hybrid #4, hybrid_rerank #4)_ **Mitigating Position Bias with Regularization for Recommender Systems**
  `2401.16427` · cs.IR · [abstract](https://arxiv.org/abs/2401.16427)

  > Fairness is a popular research topic in recent years. A research topic closely
  > related to fairness is bias and debiasing. Among different types of bias problems,
  > position bias is one of the most widely encountered symptoms. Position bias means
  > that recommended items on top of the recommendation list has a higher likelihood
  > to be clicked than items on bottom of the same list. To mitigate this problem, we
  > propose to ...

- _(dense #4)_ **Picky Eaters Make For Better Raters**
  `2401.03193` · cs.IR · [abstract](https://arxiv.org/abs/2401.03193)

  > It has been established in the literature that the number of ratings and the
  > scores restaurants obtain on online rating systems (ORS) significantly impact
  > their revenue. However, when a restaurant has a limited number of ratings, it may
  > be challenging to predict its future performance. It may well be that ratings
  > reveal more about the user who did the rating than about the quality of the
  > restaurant. This motivates ...

- _(bm25 #4)_ **Enhancing Scalability in Recommender Systems through Lottery Ticket Hypothesis and Knowledge Distillation-based Neural Network Pruning**
  `2401.10484` · cs.IR · [abstract](https://arxiv.org/abs/2401.10484)

  > This study introduces an innovative approach aimed at the efficient pruning of
  > neural networks, with a particular focus on their deployment on edge devices. Our
  > method involves the integration of the Lottery Ticket Hypothesis (LTH) with the
  > Knowledge Distillation (KD) framework, resulting in the formulation of three
  > distinct pruning models. These models have been developed to address scalability
  > issue in recommender ...

- _(hybrid_rerank #5)_ **Combining Embedding-Based and Semantic-Based Models for Post-hoc Explanations in Recommender Systems**
  `2401.04474` · cs.IR · [abstract](https://arxiv.org/abs/2401.04474)

  > In today's data-rich environment, recommender systems play a crucial role in
  > decision support systems. They provide to users personalized recommendations and
  > explanations about these recommendations. Embedding-based models, despite their
  > widespread use, often suffer from a lack of interpretability, which can undermine
  > trust and user engagement. This paper presents an approach that combines
  > embedding-based and ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 26. `q026` · vocabulary_mismatch

> **Why do chatbots sometimes make things up?**

**Labelled relevant:**

- **Navigating Uncertainty: Optimizing API Dependency for Hallucination Reduction in Closed-Book Question Answering**
  `2401.01780` · cs.CL · [abstract](https://arxiv.org/abs/2401.01780)

  > While Large Language Models (LLM) are able to accumulate and restore knowledge,
  > they are still prone to hallucination. Especially when faced with factual
  > questions, LLM cannot only rely on knowledge stored in parameters to guarantee
  > truthful and correct answers. Augmenting these models with the ability to search
  > on external information sources, such as the web, is a promising approach to
  > ground knowledge to retrieve ...

- **Physio: An LLM-Based Physiotherapy Advisor**
  `2401.01825` · cs.CL · [abstract](https://arxiv.org/abs/2401.01825)

  > The capabilities of the most recent language models have increased the interest in
  > integrating them into real-world applications. However, the fact that these models
  > generate plausible, yet incorrect text poses a constraint when considering their
  > use in several domains. Healthcare is a prime example of a domain where text-
  > generative trustworthiness is a hard requirement to safeguard patient well-being.
  > In this ...

**Why these were labelled** (drafted, not verified):

> The corpus says hallucination and plausible yet incorrect text; the query says make
> things up and chatbots, neither of which appears. A direct test of whether dense
> retrieval bridges the gap where BM25 cannot.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #1, dense #2, hybrid #1, hybrid_rerank #1)_ **Walert: Putting Conversational Search Knowledge into Action by Building and Evaluating a Large Language Model-Powered Chatbot**
  `2401.07216` · cs.IR · [abstract](https://arxiv.org/abs/2401.07216)

  > Creating and deploying customized applications is crucial for operational success
  > and enriching user experiences in the rapidly evolving modern business world. A
  > prominent facet of modern user experiences is the integration of chatbots or voice
  > assistants. The rapid evolution of Large Language Models (LLMs) has provided a
  > powerful tool to build conversational applications. We present Walert, a
  > customized LLM-based ...

- _(dense #1, hybrid #4, hybrid_rerank #3)_ **AI Revolution on Chat Bot: Evidence from a Randomized Controlled Experiment**
  `2401.10956` · cs.HC · [abstract](https://arxiv.org/abs/2401.10956)

  > In recent years, generative AI has undergone major advancements, demonstrating
  > significant promise in augmenting human productivity. Notably, large language
  > models (LLM), with ChatGPT-4 as an example, have drawn considerable attention.
  > Numerous articles have examined the impact of LLM-based tools on human
  > productivity in lab settings and designed tasks or in observational studies.
  > Despite recent advances, field ...

- _(hybrid #2, hybrid_rerank #5)_ **ChatGPT for Conversational Recommendation: Refining Recommendations by Reprompting with Feedback**
  `2401.03605` · cs.IR · [abstract](https://arxiv.org/abs/2401.03605)

  > Recommendation algorithms have been pivotal in handling the overwhelming volume of
  > online content. However, these algorithms seldom consider direct user input,
  > resulting in superficial interaction between them. Efforts have been made to
  > include the user directly in the recommendation process through conversation, but
  > these systems too have had limited interactivity. Recently, Large Language Models
  > (LLMs) like ...

- _(bm25 #2, hybrid #5)_ **Foundations of Vector Retrieval**
  `2401.09350` · cs.DS · [abstract](https://arxiv.org/abs/2401.09350)

  > Vectors are universal mathematical objects that can represent text, images,
  > speech, or a mix of these data modalities. That happens regardless of whether data
  > is represented by hand-crafted features or learnt embeddings. Collect a large
  > enough quantity of such vectors and the question of retrieval becomes urgently
  > relevant: Finding vectors that are more similar to a query vector. This monograph
  > is concerned with the ...

- _(hybrid_rerank #2)_ **Comparing Traditional and LLM-based Search for Image Geolocation**
  `2401.10184` · cs.IR · [abstract](https://arxiv.org/abs/2401.10184)

  > Web search engines have long served as indispensable tools for information
  > retrieval; user behavior and query formulation strategies have been well studied.
  > The introduction of search engines powered by large language models (LLMs)
  > suggested more conversational search and new types of query strategies. In this
  > paper, we compare traditional and LLM-based search for the task of image
  > geolocation, i.e., determining the ...

</details>

Outcome: bm25 **MISSED** · dense found · hybrid **MISSED** · hybrid_rerank **MISSED**

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 27. `q027` · vocabulary_mismatch

> **How can a computer work out whether a news website is reliable?**

**Labelled relevant:**

- **Evaluating Trustworthiness of Online News Publishers via Article Classification**
  `2401.01781` · cs.IR · [abstract](https://arxiv.org/abs/2401.01781)

  > The proliferation of low-quality online information in today's era has underscored
  > the need for robust and automatic mechanisms to evaluate the trustworthiness of
  > online news publishers. In this paper, we analyse the trustworthiness of online
  > news media outlets by leveraging a dataset of 4033 news stories from 40 different
  > sources. We aim to infer the trustworthiness level of the source based on the
  > classification ...

**Why these were labelled** (drafted, not verified):

> The corpus says trustworthiness of online news publishers; the query says reliable and
> news website. Reliable and trustworthy share no stem.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #1, dense #2, hybrid #2, hybrid_rerank #1)_ **Exploring Content-Based and Meta-Data Analysis for Detecting Fake News Infodemic: A case study on COVID-19**
  `2401.08841` · cs.IR · [abstract](https://arxiv.org/abs/2401.08841)

  > The coronavirus pandemic (COVID-19) is probably the most disruptive global health
  > disaster in recent history. It negatively impacted the whole world and virtually
  > brought the global economy to a standstill. However, as the virus was spreading,
  > infecting people and claiming thousands of lives so was the spread and propagation
  > of fake news, misinformation and disinformation about the event. These included
  > the spread ...

- _(bm25 #2, hybrid #3, hybrid_rerank #5)_ **Information Retrieval and Classification of Real-Time Multi-Source Hurricane Evacuation Notices**
  `2401.06789` · cs.IR · [abstract](https://arxiv.org/abs/2401.06789)

  > For an approaching disaster, the tracking of time-sensitive critical information
  > such as hurricane evacuation notices is challenging in the United States. These
  > notices are issued and distributed rapidly by numerous local authorities that may
  > spread across multiple states. They often undergo frequent updates and are
  > distributed through diverse online portals lacking standard formats. In this
  > study, we developed an ...

- _(bm25 #4, hybrid #4, hybrid_rerank #3)_ **Physio: An LLM-Based Physiotherapy Advisor**
  `2401.01825` · cs.CL · [abstract](https://arxiv.org/abs/2401.01825)

  > The capabilities of the most recent language models have increased the interest in
  > integrating them into real-world applications. However, the fact that these models
  > generate plausible, yet incorrect text poses a constraint when considering their
  > use in several domains. Healthcare is a prime example of a domain where text-
  > generative trustworthiness is a hard requirement to safeguard patient well-being.
  > In this ...

- _(dense #3, hybrid_rerank #4)_ **Large Language Models vs. Search Engines: Evaluating User Preferences Across Varied Information Retrieval Scenarios**
  `2401.05761` · cs.IR · [abstract](https://arxiv.org/abs/2401.05761)

  > This study embarked on a comprehensive exploration of user preferences between
  > Search Engines and Large Language Models (LLMs) in the context of various
  > information retrieval scenarios. Conducted with a sample size of 100 internet
  > users (N=100) from across the United States, the research delved into 20 distinct
  > use cases ranging from factual searches, such as looking up COVID-19 guidelines,
  > to more subjective tasks, ...

- _(dense #4)_ **Algorithmic amplification of biases on Google Search**
  `2401.09044` · cs.CY · [abstract](https://arxiv.org/abs/2401.09044)

  > The evolution of information-seeking processes, driven by search engines like
  > Google, has transformed the access to information people have. This paper
  > investigates how individuals' preexisting attitudes influence the modern
  > information-seeking process, specifically the results presented by Google Search.
  > Through a comprehensive study involving surveys and information-seeking tasks
  > focusing on the topic of abortion, ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 28. `q028` · vocabulary_mismatch

> **Working out what clothes will fit a person buying over the internet**

**Labelled relevant:**

- **Tailor: Size Recommendations for High-End Fashion Marketplaces**
  `2401.01978` · cs.IR · [abstract](https://arxiv.org/abs/2401.01978)

  > In the ever-changing and dynamic realm of high-end fashion marketplaces, providing
  > accurate and personalized size recommendations has become a critical aspect.
  > Meeting customer expectations in this regard is not only crucial for ensuring
  > their satisfaction but also plays a pivotal role in driving customer retention,
  > which is a key metric for the success of any fashion retailer. We propose a novel
  > sequence ...

**Why these were labelled** (drafted, not verified):

> The corpus says size recommendations for high-end fashion marketplaces. The query
> avoids size, recommendation and fashion entirely.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #1, hybrid #3, hybrid_rerank #5)_ **Lifelogging As An Extreme Form of Personal Information Management -- What Lessons To Learn**
  `2401.05767` · cs.IR · [abstract](https://arxiv.org/abs/2401.05767)

  > Personal data includes the digital footprints that we leave behind as part of our
  > everyday activities, both online and offline in the real world. It includes data
  > we collect ourselves, such as from wearables, as well as the data collected by
  > others about our online behaviour and activities. Sometimes we are able to use the
  > personal data we ourselves collect, in order to examine some parts of our lives
  > but for the ...

- _(bm25 #3, dense #2, hybrid #2, hybrid_rerank #4)_ **What Else Would I Like? A User Simulator using Alternatives for Improved Evaluation of Fashion Conversational Recommendation Systems**
  `2401.05783` · cs.IR · [abstract](https://arxiv.org/abs/2401.05783)

  > In Conversational Recommendation Systems (CRS), a user can provide feedback on
  > recommended items at each interaction turn, leading the CRS towards more desirable
  > recommendations. Currently, different types of CRS offer various possibilities for
  > feedback, i.e., natural language feedback, or answering clarifying questions. In
  > most cases, a user simulator is employed for training as well as evaluating the
  > CRS. Such ...

- _(dense #3, hybrid #4, hybrid_rerank #2)_ **Model-Free Approximate Bayesian Learning for Large-Scale Conversion Funnel Optimization**
  `2401.06710` · cs.LG · [abstract](https://arxiv.org/abs/2401.06710)

  > The flexibility of choosing the ad action as a function of the consumer state is
  > critical for modern-day marketing campaigns. We study the problem of identifying
  > the optimal sequential personalized interventions that maximize the adoption
  > probability for a new product. We model consumer behavior by a conversion funnel
  > that captures the state of each consumer (e.g., interaction history with the firm)
  > and allows the ...

- _(bm25 #2, hybrid #5)_ **UniRQR: A Unified Model for Retrieval Decision, Query, and Response Generation in Internet-Based Knowledge Dialogue Systems**
  `2401.06811` · cs.IR · [abstract](https://arxiv.org/abs/2401.06811)

  > Knowledge-based dialogue systems with internet retrieval have recently attracted
  > considerable attention from researchers. The dialogue systems overcome a major
  > limitation of traditional knowledge dialogue systems, where the timeliness of
  > knowledge cannot be assured, hence providing greater practical application value.
  > Knowledge-based dialogue systems with internet retrieval can be typically
  > segmented into three ...

- _(dense #5, hybrid_rerank #3)_ **Lightweight Modality Adaptation to Sequential Recommendation via Correlation Supervision**
  `2401.07257` · cs.IR · [abstract](https://arxiv.org/abs/2401.07257)

  > In Sequential Recommenders (SR), encoding and utilizing modalities in an end-to-
  > end manner is costly in terms of modality encoder sizes. Two-stage approaches can
  > mitigate such concerns, but they suffer from poor performance due to modality
  > forgetting, where the sequential objective overshadows modality representation. We
  > propose a lightweight knowledge distillation solution that preserves both merits:
  > retaining ...

</details>

Outcome: bm25 found · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 29. `q029` · vocabulary_mismatch

> **Teaching software to put search results in a better order by observing what people click**

**Labelled relevant:**

- **Learning-to-Rank with Nested Feedback**
  `2401.04053` · cs.IR · [abstract](https://arxiv.org/abs/2401.04053)

  > Many platforms on the web present ranked lists of content to users, typically
  > optimized for engagement-, satisfaction- or retention- driven metrics. Advances in
  > the Learning-to-Rank (LTR) research literature have enabled rapid growth in this
  > application area. Several popular interfaces now include nested lists, where users
  > can enter a 2nd-level feed via any given 1st-level item. Naturally, this has
  > implications for ...

**Why these were labelled** (drafted, not verified):

> The corpus says learning-to-rank and nested feedback; the query describes the
> mechanism in plain words. Note that order and rank are distinct terms lexically.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(dense #1, hybrid #4)_ **On the Influence of Reading Sequences on Knowledge Gain during Web Search**
  `2401.05148` · cs.IR · [abstract](https://arxiv.org/abs/2401.05148)

  > Nowadays, learning increasingly involves the usage of search engines and web
  > resources. The related interdisciplinary research field search as learning aims to
  > understand how people learn on the web. Previous work has investigated several
  > feature classes to predict, for instance, the expected knowledge gain during web
  > search. Therein, eye-tracking features have not been extensively studied so far.
  > In this paper, we ...

- _(bm25 #3, hybrid_rerank #1)_ **Lifelogging As An Extreme Form of Personal Information Management -- What Lessons To Learn**
  `2401.05767` · cs.IR · [abstract](https://arxiv.org/abs/2401.05767)

  > Personal data includes the digital footprints that we leave behind as part of our
  > everyday activities, both online and offline in the real world. It includes data
  > we collect ourselves, such as from wearables, as well as the data collected by
  > others about our online behaviour and activities. Sometimes we are able to use the
  > personal data we ourselves collect, in order to examine some parts of our lives
  > but for the ...

- _(bm25 #2, dense #3, hybrid #1, hybrid_rerank #2)_ **Ranking Heterogeneous Search Result Pages using the Interactive Probability Ranking Principle**
  `2401.08267` · cs.IR · [abstract](https://arxiv.org/abs/2401.08267)

  > The Probability Ranking Principle (PRP) ranks search results based on their
  > expected utility derived solely from document contents, often overlooking the
  > nuances of presentation and user interaction. However, with the evolution of
  > Search Engine Result Pages (SERPs), now comprising a variety of result cards, the
  > manner in which these results are presented is pivotal in influencing user
  > engagement and satisfaction. ...

- _(bm25 #1, hybrid #3, hybrid_rerank #5)_ **Algorithmic amplification of biases on Google Search**
  `2401.09044` · cs.CY · [abstract](https://arxiv.org/abs/2401.09044)

  > The evolution of information-seeking processes, driven by search engines like
  > Google, has transformed the access to information people have. This paper
  > investigates how individuals' preexisting attitudes influence the modern
  > information-seeking process, specifically the results presented by Google Search.
  > Through a comprehensive study involving surveys and information-seeking tasks
  > focusing on the topic of abortion, ...

- _(bm25 #4, dense #5, hybrid #2, hybrid_rerank #3)_ **Comparing Traditional and LLM-based Search for Image Geolocation**
  `2401.10184` · cs.IR · [abstract](https://arxiv.org/abs/2401.10184)

  > Web search engines have long served as indispensable tools for information
  > retrieval; user behavior and query formulation strategies have been well studied.
  > The introduction of search engines powered by large language models (LLMs)
  > suggested more conversational search and new types of query strategies. In this
  > paper, we compare traditional and LLM-based search for the task of image
  > geolocation, i.e., determining the ...

</details>

Outcome: bm25 **MISSED** · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## 30. `q030` · vocabulary_mismatch

> **Somebody deliberately feeding false information into a shopping site's suggestion engine**

**Labelled relevant:**

- **Poisoning Attacks against Recommender Systems: A Survey**
  `2401.01527` · cs.IR · [abstract](https://arxiv.org/abs/2401.01527)

  > Modern recommender systems (RS) have seen substantial success, yet they remain
  > vulnerable to malicious activities, notably poisoning attacks. These attacks
  > involve injecting malicious data into the training datasets of RS, thereby
  > compromising their integrity and manipulating recommendation outcomes for gaining
  > illicit profits. This survey paper provides a systematic and up-to-date review of
  > the research landscape ...

**Why these were labelled** (drafted, not verified):

> The corpus says poisoning attacks against recommender systems; the query says feeding
> false information and suggestion engine. Tests paraphrase of both the attack and the
> system.

<details>
<summary><b>Retrieved but not labelled</b> — check whether any of these should be relevant</summary>

- _(bm25 #1, dense #2, hybrid #1)_ **Large Language Models vs. Search Engines: Evaluating User Preferences Across Varied Information Retrieval Scenarios**
  `2401.05761` · cs.IR · [abstract](https://arxiv.org/abs/2401.05761)

  > This study embarked on a comprehensive exploration of user preferences between
  > Search Engines and Large Language Models (LLMs) in the context of various
  > information retrieval scenarios. Conducted with a sample size of 100 internet
  > users (N=100) from across the United States, the research delved into 20 distinct
  > use cases ranging from factual searches, such as looking up COVID-19 guidelines,
  > to more subjective tasks, ...

- _(bm25 #2, hybrid #2, hybrid_rerank #1)_ **Exploring Content-Based and Meta-Data Analysis for Detecting Fake News Infodemic: A case study on COVID-19**
  `2401.08841` · cs.IR · [abstract](https://arxiv.org/abs/2401.08841)

  > The coronavirus pandemic (COVID-19) is probably the most disruptive global health
  > disaster in recent history. It negatively impacted the whole world and virtually
  > brought the global economy to a standstill. However, as the virus was spreading,
  > infecting people and claiming thousands of lives so was the spread and propagation
  > of fake news, misinformation and disinformation about the event. These included
  > the spread ...

- _(hybrid_rerank #2)_ **Content-Aware Tweet Location Inference using Quadtree Spatial Partitioning and Jaccard-Cosine Word Embedding**
  `2401.08506` · cs.IR · [abstract](https://arxiv.org/abs/2401.08506)

  > Inferring locations from user texts on social media platforms is a non-trivial and
  > challenging problem relating to public safety. We propose a novel non-uniform
  > grid-based approach for location inference from Twitter messages using Quadtree
  > spatial partitions. The proposed algorithm uses natural language processing (NLP)
  > for semantic understanding and incorporates Cosine similarity and Jaccard
  > similarity measures ...

- _(dense #3)_ **ChatGPT for Conversational Recommendation: Refining Recommendations by Reprompting with Feedback**
  `2401.03605` · cs.IR · [abstract](https://arxiv.org/abs/2401.03605)

  > Recommendation algorithms have been pivotal in handling the overwhelming volume of
  > online content. However, these algorithms seldom consider direct user input,
  > resulting in superficial interaction between them. Efforts have been made to
  > include the user directly in the recommendation process through conversation, but
  > these systems too have had limited interactivity. Recently, Large Language Models
  > (LLMs) like ...

- _(bm25 #3, hybrid #5, hybrid_rerank #5)_ **LLMRS: Unlocking Potentials of LLM-Based Recommender Systems for Software Purchase**
  `2401.06676` · cs.IR · [abstract](https://arxiv.org/abs/2401.06676)

  > Recommendation systems are ubiquitous, from Spotify playlist suggestions to Amazon
  > product suggestions. Nevertheless, depending on the methodology or the dataset,
  > these systems typically fail to capture user preferences and generate general
  > recommendations. Recent advancements in Large Language Models (LLM) offer
  > promising results for analyzing user queries. However, employing these models to
  > capture user ...

</details>

Outcome: bm25 **MISSED** · dense found · hybrid found · hybrid_rerank found

```
VERDICT:
REMOVE:
ADD:
REWORD:
```

---

## When you are done

Tell me the corrections and I will apply them to the dataset, flip `reviewed`
to true on the examples you confirmed, change `label_source` to
`human_verified`, and re-run the experiments. The re-run decides whether the
dense-over-hybrid result survives review, which is what ADR-0008 says is needed
before the production default changes.

Partial review is still useful. Twenty reviewed questions with honest labels
are worth more than thirty drafted ones, and the review status is reported per
example rather than as a single flag precisely so a partial review can be
recorded accurately.
