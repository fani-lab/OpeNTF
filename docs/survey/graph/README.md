This repository contains supplementary materials for our published survey paper:


<h1 align="center">
  <a href="https://[example.com](https://hosseinfani.github.io/res/papers/2025_CSUR_A_Survey_of_Subgraph_Optimization_for_Expert_Team_Formation.pdf)"><strong>A Survey of Subgraph Optimization for Expert Team Formation</strong></a>
</h1>

# Journal Highlight

[**ACM Computing Surveys**](https://dl.acm.org/journal/csur)

**Impact Factor:** 23.8

**H-Index:** 232

**Ranking:** 1st out of 143 journals in Computer Science – Theory & Methods

<h2 style="border-bottom: none;">Timeline</h2>
________________________________________________________________________________________________________________________________________________________

Expert Team Formation is the search for gathering a team of experts who are expected to collaboratively work towards accomplishing a given project, a problem that has historically been solved in a variety of ways, including manually in a time-consuming and bias-filled manner, and algorithmically within disciplines like social sciences and management. In the present effort, while providing a taxonomy to distinguish between search-based versus learning-based approaches, we survey graph-based studies from the search-based category, motivated as they comprise the mainstream. We present a unifying and vetted overview of the various definitions in this realm, scrutinize assumptions, and identify shortfalls. We start by reviewing initial approaches to the Expert Team Formation problem to lay the conceptual foundations and set forth the necessary notions for a more grounded view of this realm. Next, we provide a detailed view of graph-based Expert Team Formation approaches based on the objective functions they optimize. We lay out who builds on whom and how algorithms have evolved to solve the drawbacks of previous works. Further, we categorize evaluation schemas and elaborate on metrics and insights that can be drawn from each. Referring to the evaluation schemas and metrics, we compare works and propose future directions.

Within the search-based category, operations research, by and large, optimizes the mutually independent selection of experts, overlooking  the organizational and collaborative ties among individuals [^1][^4][^5][^6]. However, graph-based methods rely on the premise that a team is inherently relational and is a property of the interaction among the experts and how effectively they can collaborate. Moreover,
learning-based methods have been proposed to bring efficiency while enhancing efficacy due to their iterative and online learning procedure [^7][^8]. As seen in Figure 1, there is an overlap between team formation approaches. As an instance, graph-based and operations research-based methods, where optimization functions have been defined based on linear or nonlinear equation of Boolean variables 
representing edges on the expert graph [^9]. Additionally, learning-based methods utilize the expert graph to learn vector representations of skills using graph neural networks (GNNs), which helps reduce the complexity of neural models at the input layer [^12]. Specifically, graph neural networks [^10][^11] provide an effective and efficient general framework for solving graph analytics problems by converting 
a graph into a low-dimensional vector space while preserving its graph-structured information. Having demonstrated strong performances across a wide range of problems, including natural language processing [^13], knowledge graphs [^14] and recommender systems [^15], graph neural networks are gradually finding their application in Expert Team Formation [^12]. 

<p align="center">
<img src="figures/figs_draft/venn.png" alt="Figure1: Overlapping Team Formation Approaches">
</p>

This survey pertains to the graph-based Expert Team Formation algorithms, that is, those that employ graphs to model the experts’ collaboration ties followed by subgraph optimization algorithms, as they comprise the mainstream body of research. While some OR-based works model the dataset as a graph structure, they opt for linear/non-linear integer/real programming methods as opposed to subgraph optimization. In this survey, we include works that not only model the data as graphs but also apply subgraph optimization methods. We exclude works that are based on operations research and learning-based methods, as they differ fundamentally from subgraph optimization algorithms. We recognize the importance of these areas and the wealth of work they include, but a thorough analysis of them is beyond the scope of this work and merits separate surveys.



We present a comprehensive overview of 18 seminal graph-based solutions to the Team Formation problem, 
including 13 proposed optimization objectives, after screening 63 algorithms from 126 papers. 
The examined papers in this survey can be categorized as follows:

[^1]: M. Muniz et al. A column generation approach for the team formation problem. https://www.sciencedirect.com/science/article/pii/S0305054823002708.
[^2]: A. Anagnostopoulos et al. Power in unity: forming teams in large-scale community systems. https://dl.acm.org/doi/pdf/10.1145/1871437.1871515.
[^3]: E. Fitzpatrick et al. Forming effective worker teams with multi-functional skill requirements. https://www.sciencedirect.com/science/article/pii/S0360835204002049.
[^4]: E. H. Durfee et al. Using hybrid scheduling for the semi-autonomous formation of expert teams. https://www.sciencedirect.com/science/article/pii/S0167739X1300068X.
[^5]: S. J. Kalayathankal et al. A modified fuzzy approach to project team selection. https://www.sciencedirect.com/science/article/pii/S2666222121000022. 
[^6]: L. Wang et al. Team Recommendation Using Order-Based Fuzzy Integral and NSGA-II in StarCraft. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9044841.
[^7]: R. Hamidi Rad et al. PyTFL: A Python-based Neural Team Formation Toolkit. https://dl.acm.org/doi/pdf/10.1145/3459637.3481992.
[^8]: A. Dashti et al. Effective Neural Team Formation via Negative Samples. https://dl.acm.org/doi/pdf/10.1145/3511808.3557590.
[^9]: M. B. Campelo et al. The sociotechnical teams formation problem: a mathematical optimization approach. https://link.springer.com/content/pdf/10.1007/s10479-018-2759-5.pdf.
[^10]: R. Bing et al. Heterogeneous graph neural networks analysis: a survey of techniques. https://link.springer.com/content/pdf/10.1007/s10462-022-10375-2.pdf. 
[^11]: Z. Wu et al. A Comprehensive Survey on Graph Neural Networks. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9046288.
[^12]: H. Nguyen et al. Learning heterogeneous subgraph representations for team discovery. https://link.springer.com/content/pdf/10.1007/s10791-023-09421-6.pdf.
[^13]: L. Wu et al. Graph Neural Networks for Natural Language Processing: A Survey. https://www.nowpublishers.com/article/Details/MAL-096.
[^14]: Z. Ye et al. A Comprehensive Survey of Graph Neural Networks for Knowledge Graphs. https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9831453.
[^15]: S. Wu et al. Graph Neural Networks in Recommender Systems: A Survey. https://dl.acm.org/doi/pdf/10.1145/3535101.
```




─categirized_papers
      ├───capacity_of_team_members
      ├───constraint
      │   ├───authority
      │   ├───communication_cost
      │   │   ├───buttleneck
      │   │   ├───dense
      │   │   ├───diameter
      │   │   ├───graph_clustering
      │   │   ├───stainertree
      │   │   └───sum_of_edge_weight
      │   ├───geographical proximity
      │   ├───trust
      │   └───workload
      ├───dynamic_network
      ├───efficiency
      ├───fairness
      ├───grouped_team
      ├───keyword_search_and_community_search
      ├───learning_based
      │   ├───game_theory
      │   └───learning_search_based
      ├───multi_objectiver
      ├───number_of_created_teams
      │   ├───more_than _one
      │   │   ├───pareto_set
      │   │   └───top_k
      │   └───one
      ├───old
      │   ├───education
      │   ├───engineering
      │   ├───multi_skill_heuristic_solution
      │   ├───network
      │   ├───performance
      │   └───team_member_characteristic
      ├───operation_research
      │   ├───fuzzy
      │   ├───genetic_algorithm
      │   ├───hierarchical
      │   ├───integer_programming
      │   └───linear_programming
      ├───similarity_between_two_graphs
      │   ├───graph_pattern
      │   └───kernel_replacing_a_member
      ├───surveys
      ├───team with leader
      └───team_size
          ├───at_least_k_person_for_each_skill
          ├───at_most_k_responsiblity_for_each_person
          └───small_teams
```







