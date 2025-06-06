This repository contains supplementary materials for our published survey paper:


<h1 align="center" style="border-bottom: none;">
  <a href="https://[example.com](https://hosseinfani.github.io/res/papers/2025_CSUR_A_Survey_of_Subgraph_Optimization_for_Expert_Team_Formation.pdf)"><strong>A Survey of Subgraph Optimization for Expert Team Formation</strong></a>
</h1>

![#25D9C8](https://placehold.co/15x15/25D9C8/25D9C8.png) **Journal Highlight** 

> **[ACM Computing Surveys](https://dl.acm.org/journal/csur):**
      
    
> **Impact Factor:**
23.8

> **H-Index:**
232

> **Ranking:**
1st out of 143 journals in Computer Science – Theory & Methods   

  
![#25D9C8](https://placehold.co/15x15/25D9C8/25D9C8.png) **Timeline**  

> **Started:**
June 5, 2021


> **Submitted:**
November 22, 2023

> **[First Round Review:](https://hosseinfani.github.io/res/papers/2025_CSUR_R1_A_Survey_of_Subgraph_Optimization_for_Expert_Team_Formation.pdf)**
October 11, 2024

> **[Second Round Review:](https://hosseinfani.github.io/res/papers/2025_CSUR_R2_A_Survey_of_Subgraph_Optimization_for_Expert_Team_Formation.pdf)**
March 1, 2025

> **[Notification of Acceptance:](https://hosseinfani.github.io/res/papers/2025_CSUR_R3_A_Survey_of_Subgraph_Optimization_for_Expert_Team_Formation.txt)**
May 8, 2025


________________________________________________________________________________________________________________________________________________________

Expert Team Formation refers to the task of gathering a group of experts to collaboratively complete a given project. Traditionally, this problem has been tackled either manually—often time-consuming and prone to bias—or algorithmically, within disciplines like social sciences and management. In this work, we focus on graph-based studies within the search-based category of approaches, distinguishing them from learning-based methods through a high-level taxonomy. Our goal is to provide a unified and vetted overview of existing definitions, highlight underlying assumptions, and point out current limitations. We start by reviewing initial approaches to the Expert Team Formation problem to lay the conceptual foundations and set forth the necessary notions for a more grounded view of this realm. Then, we explore how graph-based methods define and optimize different objective functions, tracing how newer algorithms build upon and improve earlier ones. Finally, we compare different approaches and outline promising directions for future work.



Team formation involves selecting experts with certain skills to form a successful task-oriented team. Team formation approaches can be distinguished based on the way optimizations are performed: i) search-based, where the search for an *almost surely* successful team (optimum team) is carried out over the subgraphs of an expert graph using subgraph optimization methods, or it is performed on subsets of experts as variants of the set cover problem [[1](https://www.sciencedirect.com/science/article/pii/S0305054823002708.), [2](https://dl.acm.org/doi/pdf/10.1145/1871437.1871515), [3](https://www.sciencedirect.com/science/article/pii/S0360835204002049.)]  using operations research (OR) techniques including integer linear/nonlinear programming, and ii) learning-based, where machine learning approaches are used to learn the distributions of experts and skills in the context of previous (un)successful teams in order to draw
future successful teams.

<div align="center">
  <img src="figures/figs_draft/methods-survey.png" alt="Figure1: A taxonomy of the computational Expert Team Formation methods." width="800" >
  <p align="center"><em>Figure1: A taxonomy of the computational Expert Team Formation methods.</em></p>
</div>

Within the search-based category, operations research, by and large, optimizes the mutually independent selection of experts, overlooking  the organizational and collaborative ties among individuals [[1](https://www.sciencedirect.com/science/article/pii/S0305054823002708.), [4](https://www.sciencedirect.com/science/article/pii/S0167739X1300068X.), [5](https://www.sciencedirect.com/science/article/pii/S2666222121000022. ), [6](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9044841.)]. However, graph-based methods rely on the premise that a team is inherently relational and is a property of the interaction among the experts and how effectively they can collaborate. Moreover, learning-based methods have been proposed to bring efficiency while enhancing efficacy due to their iterative and online learning procedure [[7](https://dl.acm.org/doi/pdf/10.1145/3459637.3481992.), [8](https://dl.acm.org/doi/pdf/10.1145/3511808.3557590)]. As seen in Figure 1, there is an overlap between team formation approaches. As an instance, graph-based and operations research-based methods, where optimization functions have been defined based on linear or nonlinear equation of Boolean variables representing edges on the expert graph [[9](https://link.springer.com/content/pdf/10.1007/s10479-018-2759-5.pdf)]. 

From Figure 2, recently, a paradigm shift to machine learning-based methods, including artificial neural networks and graph neural networks [[10](https://link.springer.com/content/pdf/10.1007/s10791-023-09421-6.pdf), [11](https://link.springer.com/chapter/10.1007/978-981-96-0567-5_15), [12](https://link.springer.com/chapter/10.1007/978-3-031-56027-9_20 ), [13](https://dl.acm.org/doi/pdf/10.1145/3701551.3703574), [14](https://hosseinfani.github.io/res/papers/2025_SIGIR_Translative_Neural_Team_Recommendation_From_Multilabel_Classification_to_Sequence_Prediction.pdf)], has been observed due to advanced hardware computing power, especially graphic processing units (GPUs) and tensor processing units (TPUs), that reduced elapsed time from months to days and/or hours opening doors to the analysis of massive collections of candidates coming from different fields. These methods are different from search-based solutions in that they *learn* the inherent structure of the ties among candidates and their skills. Wherein, all past successful and *un*successful team compositions are considered as training samples to predict future teams and the team's performance. Learning-based methods bring efficiency while enhancing efficacy due to the inherently *iterative* and *online* learning procedure, and can address the limitations of search-based solutions with respect to scalability. This line of research started with Sapienza et al. [[18](https://doi.org/10.3389/fdata.2019.00014)] who employed a non-variational autoencoder neural network architecture and is being followed by researchers through other neural-based architectures such as variational Bayesian neural network [[8](https://dl.acm.org/doi/pdf/10.1145/3511808.3557590), [7](https://dl.acm.org/doi/pdf/10.1145/3459637.3481992), [15](https://dl.acm.org/doi/pdf/10.1145/3340531.3412140), [16](https://dl.acm.org/doi/pdf/10.1145/3589762), [17](https://proceedings.mlr.press/v139/liu21m/liu21m.pdf)].

Although learning-based literature is novel, it is still in its early stages. We chose to survey the graph-based approaches as they comprise the mainstream and a relatively larger body of research to be investigated in a survey such as the present one. Further, this body of work serves as a prelude to the learning-based works, especially the ones based on graph neural network in terms of the Expert Team Formation problem definition as well as experimental and evaluation settings. We are however tracing the learning-based paradigm's literature and its progress for a future potential survey.

<p align="center">
<img src="figures/figs_draft/alluvial/alluvial_new.png" alt="Figure2: Expert Team Formation methods in time;.">
  <p align="center"><em>Figure2: Expert Team Formation methods in time;.</em></p>
</p>

This survey pertains to the graph-based Expert Team Formation algorithms, that is, those that employ graphs to model the experts’ collaboration ties followed by subgraph optimization algorithms, as they comprise the mainstream body of research. While some OR-based works model the dataset as a graph structure, they opt for linear/non-linear integer/real programming methods as opposed to subgraph optimization. In this survey, we include works that not only model the data as graphs but also apply subgraph optimization methods. We exclude works that are based on operations research and learning-based methods, as they differ fundamentally from subgraph optimization algorithms. We recognize the importance of these areas and the wealth of work they include, but a thorough analysis of them is beyond the scope of this work and merits separate surveys.

<p align="center">
<img src="figures/figs_draft/venn.png" alt="Figure3: Overlapping Team Formation Approaches.">
  <p align="center"><em>Figure3: Overlapping Team Formation Approaches.</em></p>
</p>


After screening 126 papers addressing Team Formation problems using computational and non-computational models, we present a comprehensive overview of 17 seminal graph-based research papers on the Expert Team Formation problem within the scope of our survey, including 18 unique objectives, to be optimized via 63 subgraph optimization algorithms considering variations of their exact algorithms and the heuristics they used to address the efficiency of their algorithms. Our survey brings forth a unifying and vetted methodology to review the various definitions of the notions, criticizes assumptions and comparative benchmarks, and points out shortfalls to smooth the path for future research directions. It targets the information retrieval (IR) and recommender systems (RecSys) research communities to propose new Expert Team Formation solutions and evaluate their effectiveness compared to existing methods and on datasets from various domains. Further, having regard to the unified comparative analysis, organizations and practitioners can compare different models and readily pick the most suitable one for their application to form teams of experts whose success is almost surely guaranteed.

The examined papers in this survey can be categorized as follows:

 - **team formation applications**
      - **group learning**
        - [2025--Leveraging Multicriteria Integer Programming Optimization for Effective Team Formation](https://doi.org/10.1109/tlt.2024.3401734)
        - [2023--An Integer Linear Programming Model for Team Formation in the Classroom with Constraints](https://link.springer.com/chapter/10.1007/978-3-031-40725-3_34)
        - [2014--A Method for Group Formation Using Genetic Algorithm](https://www.researchgate.net/profile/Azman-Yasin/publication/229035954_A_Method_for_Group_Formation_Using_Genetic_Algorithm/links/0deec53852db22592c000000/A-Method-for-Group-Formation-Using-Genetic-Algorithm.pdf)
        - [2013--Support group formation for users with depression in social networks](https://onlinelibrary.wiley.com/doi/full/10.1111/exsy.12038)
        - [2012--A genetic algorithm approach for group formation in collaborative learning considering multiple student characteristics](https://www.sciencedirect.com/science/article/pii/S0360131511002284)
        - [2010--Forming Reasonably Optimal Groups (FROG)](https://dl.acm.org/doi/pdf/10.1145/1880071.1880094)
      - **health care**
        - [2025--Support group formation for users with depression in social networks](https://www.sciencedirect.com/science/article/pii/S0957417425007298)
        - [2023--Understanding knowledge leadership in improving team outcomes in the health sector: a Covid-19 study](https://www.emerald.com/insight/content/doi/10.1108/bpmj-08-2022-0386/full/pdf)
      - **peer review**
        - [2025--Peer review expert group recommendation: A multi-subject coverage-based approach](https://doi.org/10.1016/j.eswa.2024.125971)
        - [2024--Reviewerly: Modeling the Reviewer Assignment Task as an Information Retrieval Problem](https://doi.org/10.1145/3627673.3679081)
      - **reviewer assignment**
        - [2024--A System and Method for Recommending Expert Reviewers for Performing Quality Assessment of an Electronic Work](https://patents.google.com/patent/US20240005230A1/en)
        - [2024--Reviewer Assignment Decision Support in an Academic Journal based on Multicriteria Assessment and Text Mining](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=10139187)
        - [2023--Reviewer Assignment Problem: A Systematic Review of the Literature](https://dl.acm.org/doi/pdf/10.1613/jair.1.14318)
        - [2023--A One-Size-Fits-All Approach to Improving Randomness in Paper Assignment](https://proceedings.neurips.cc/paper_files/paper/2023/file/2e9f9cde1b709281a06dd14f679e4c51-Paper-Conference.pdf)
        - [2023--Counterfactual Evaluation of Peer-Review Assignment Policies](https://proceedings.neurips.cc/paper_files/paper/2023/file/b7d795e655c1463d7299688d489e8ef4-Paper-Conference.pdf)
        - [2023--Group Fairness in Peer Review](https://proceedings.neurips.cc/paper_files/paper/2023/file/ccba10dd4e80e7276054222bb95d467c-Paper-Conference.pdf)
- **team formation constraints**
  - **authority**
    - [2021--A unified framework for effective team formation in social networks](https://doi.org/10.1016/j.eswa.2021.114886)
    - [2018--Effective Team Formation in Expert Networks](https://ceur-ws.org/Vol-2100/paper4.pdf)
    - [2017--Authority-Based Team Discovery in Social Networks](https://arxiv.org/pdf/1611.02992)
  - **communication cost**
    - **dense subgraph**
      - [2018--A multi-objective formulation of the team formation problem in social networks: preliminary results](https://doi.org/10.1145/3205455.3205634)
      - [2017--TeamGen: An Interactive Team Formation System Based on Professional Social Network](https://doi.org/10.1145/3041021.3054725)
      - [2013--Towards realistic team formation in social networks based on densest subgraphs](https://doi.org/10.1145/2488388.2488482)
      - [2012--Multi-skill Collaborative Teams based on Densest Subgraphs](https://doi.org/10.1137/1.9781611972825.15)
      - [2011--Interaction mining and skill-dependent recommendations for multi-objective team composition](https://doi.org/10.1016/j.datak.2011.06.004)
      - [2010--Team Formation for Generalized Tasks in Expertise Social Networks](https://doi.org/10.1109/socialcom.2010.12)
    - **diameter**
      - [2012--Online team formation in social networks](https://doi.org/10.1145/2187836.2187950)
      - [2012--Multi-skill Collaborative Teams based on Densest Subgraphs](https://doi.org/10.1137/1.9781611972825.15)
      - [2012--Capacitated Team Formation Problem on Social Networks](https://doi.org/10.1145/2339530.2339690)
      - [2009--Finding a team of experts in social networks](https://doi.org/10.1145/1557019.1557074)
    - **steiner tree**
      - [2012--Online Team Formation in Social Networks](https://doi.org/10.1145/2187836.2187950)
      - [2012--Capacitated Team Formation Problem on Social Networks](https://doi.org/10.1145/2339530.2339690)
      - [2010--Team Formation for Generalized Tasks](https://doi.org/10.1109/socialcom.2010.12)
      - [2009--Finding a Team of Experts in Social Networks](https://doi.org/10.1145/1557019.1557074)
    - **sum of edge weight**
      - [2021--A unified framework for effective team formation in social networks](https://doi.org/10.1016/j.eswa.2021.114886)
      - [2019--Optimized group formation for solving collaborative tasks](https://doi.org/10.1007/s00778-018-0516-7)
      - [2016--Authority-Based Team Discovery in Social Networks](https://doi.org/10.1137/1.9781611972818.45)
      - [2016--Forming Grouped Teams with Efficient Collaboration in Social Networks](https://doi.org/10.1093/comjnl/bxw088)
      - [2011--Discovering top-k teams of experts with/without a leader in social networks](https://doi.org/10.1145/2063576.2063718)   
  - **geographical proximiti**
    - [2021--A unified framework for effective team formation in social networks](https://doi.org/10.1016/j.eswa.2021.114886)
    - [2020--Negotiating team formation using deep reinforcement learning](https://doi.org/10.1016/j.artint.2020.103356)
    - [2013--Towards realistic team formation in social networks based on densest subgraphs](https://doi.org/10.1145/2488388.2488482)
  - **trust**
    - [2021--A unified framework for effective team formation in social networks](https://doi.org/10.1016/j.eswa.2021.114886)
  - **workload**
    - [2012--Online Team Formation in Social Networks](https://doi.org/10.1145/2187836.2187950)
- **dynamic network**
    - [2024--A Streaming Approach to Neural Team Formation Training](https://link.springer.com/chapter/10.1007/978-3-031-56027-9_20)
    - [2021--A unified framework for effective team formation in social networks](https://doi.org/10.1016/j.eswa.2021.114886)
    - [2020--Efficient Team Formation in Social Networks based on Constrained Pattern Graph](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9101508)
    - [2018--Graph Pattern Matching for Dynamic Team Formation](https://arxiv.org/pdf/1801.01012)
    - [2015--Replacing the Irreplaceable: Fast Algorithms for Team Member Recommendation](https://doi.org/10.1145/2736277.2741132)    
- **efficiency**
    - [2020--Efficient Team Formation in Social Networks based on Constrained Pattern Graph](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9101508)
    - [2018--Effective Team Formation in Expert Networks](https://ceur-ws.org/Vol-2100/paper4.pdf)
    - [2017--TeamGen: An Interactive Team Formation System Based on Professional Social Network](https://doi.org/10.1145/3041021.3054725)
    - [2016--Top-k Team Recommendation in Spatial Crowdsourcing](https://doi.org/10.1007/978-3-319-39937-9_15)
    - [2016--Forming Grouped Teams with Efficient Collaboration in Social Networks](https://doi.org/10.1093/comjnl/bxw088)
    - [2015--Replacing the Irreplaceable: Fast Algorithms for Team Member Recommendation](https://doi.org/10.1145/2736277.2741132)   
    - [2014--Two-Phase Pareto Set Discovery for Team Formation in Social Networks](https://doi.org/10.1109/wi-iat.2014.112)
    - [2012--Online Team Formation in Social Networks](https://doi.org/10.1145/2187836.2187950)
    - [2012--Capacitated Team Formation Problem on Social Networks](https://doi.org/10.1145/2339530.2339690)
- **grouped team**
    - [2019--Optimized group formation for solving collaborative tasks](https://doi.org/10.1007/s00778-018-0516-7)
    - [2016--Forming Grouped Teams with Efficient Collaboration in Social Networks](https://doi.org/10.1093/comjnl/bxw088)
    - [2010--Team Formation for Generalized Tasks in Expertise Social Networks](https://doi.org/10.1109/socialcom.2010.12)
- **history-old**
    - **education**
        - [2010--Forming reasonably optimal groups (FROG)](https://doi.org/10.1145/1880071.1880094)
    - **engineering**
        - [2005--An Integrated Methodological Framework for Project Task Coordination and Team Organization in Concurrent Engineering](https://journals.sagepub.com/doi/pdf/10.1177/1063293X05056462)
        - [2004--Modeling team member characteristics for the formation of a multifunctional team in concurrent engineering](https://doi.org/10.1109/tem.2004.826011)
    - **multi-skill heuristic solution**
        - [2016--Forming Grouped Teams with Efficient Collaboration in Social Networks](https://doi.org/10.1093/comjnl/bxw088)
        - [2004--Modeling team member characteristics for the formation of a multifunctional team in concurrent engineering](https://doi.org/10.1109/tem.2004.826011)
        - [2004--Forming effective worker teams with multi-functional skill requirements](https://doi.org/10.1016/j.cie.2004.12.014)
    - **network**
        - [2010--A Multi-level Study of Free-Loading in Dynamic Groups: The Importance of Initial Network Topology](https://doi.org/10.1109/incos.2010.96)
        - [2010--The Small-World Effect: The Influence of Macro-Level Properties of Developer Collaboration Networks on Open-Source Project Success](https://dl.acm.org/doi/pdf/10.1145/1824760.1824763)
        - [2006--Application of Social Network Analysis to Collaborative Team Formation](https://doi.org/10.1109/cts.2006.18)
        - [2004--Adapting Network Structure for Efficient Team Formation](https://cdn.aaai.org/Symposia/Fall/2004/FS-04-02/FS04-02-001.pdf)
        - [2003--Team Formation in Complex Networks](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=cf6f04abe0ffb6d994592204ab8bcfc8606f3120)
    - **Performance**
        - [2019--Factors affecting team performance in it sector: An exploratory analysis](https://gitarattan.edu.in/wp-content/uploads/2020/05/Ch-01.pdf)
        - [2005--An Integrated Methodological Framework for Project Task Coordination and Team Organization in Concurrent Engineering](https://journals.sagepub.com/doi/pdf/10.1177/1063293X05056462)
        - [1993--The Wisdom of Teams](https://www.abhe.org/wp-content/uploads/2025/02/Phil_Orazi_Wisdom-of-Teams-Abstract.pdf)
        - [1992--The Role of Mental Models in Team Performance in Complex Systems](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=199457)
    - **team member characteristic**
        - [2004--Modeling team member characteristics for the formation of a multifunctional team in concurrent engineering](https://doi.org/10.1109/tem.2004.826011)
    - **other**
        - [2008--On Teams, Teamwork, and Team Performance: Discoveries and Developments](https://journals.sagepub.com/doi/pdf/10.1518/001872008X288457)
        - [2001--Using student conative behaviors and technical skills to form effective project teams](https://doi.org/10.1109/fie.2001.964039)
        - [2000--Teamwork in multi-person systems: a review and analysis](https://doi.org/10.1080/00140130050084879)
        - [1998--Forming teams: an analytical approach](https://doi.org/10.1080/07408179908969808)
        - [1997--toward flexible teemworks](https://www.jair.org/index.php/jair/article/view/10193/24210)
        - [1997--On Team Formation](https://www.researchgate.net/profile/Philip-Cohen-8/publication/2238935_On_Team_Formation/links/56a658ac08ae2c689d39e796/On-Team-Formation.pdf)
        - [1965--Developmental sequence in small groups.](https://web.mit.edu/curhan/www/docs/Articles/15341_Readings/Group_Dynamics/Tuckman_1965_Developmental_sequence_in_small_groups.pdf)
- **keyword search and community discovery**
    - [2020--Effective Keyword Search Over Weighted Graphs](https://doi.org/10.1109/tkde.2020.2985376)
    - [2019--Compact group discovery in attributed graphs and social networks](https://doi.org/10.1016/j.ipm.2019.102054)
    - [2019--Robust keyword search in large attributed graphs](https://doi.org/10.1007/s10791-020-09379-9)
    - [2018--Multityped Community Discovery in Time-Evolving Heterogeneous Information Networks Based on Tensor Decomposition](https://doi.org/10.1155/2018/9653404)
    - [2016--Effective community search for large attributed graphs](https://doi.org/10.14778/2994509.2994538)
    - [2011--Keyword Search in Graphs: Finding r-cliques](https://dl.acm.org/doi/pdf/10.14778/2021017.2021025)
    - [2010--Composite recommendations: from items to packages](https://doi.org/10.1007/s11704-012-2014-1)
    - [2009--Querying Communities in Relational Databases](https://doi.org/10.1109/icde.2009.67)
    - [2008--Keyword proximity search in complex data graphs](https://doi.org/10.1145/1376616.1376708)  
- **learning-based**
  - **main**
    - [2025--A QUBO Framework for Team Formation](https://arxiv.org/pdf/2503.23209)
    - [2024--A Streaming Approach to Neural Team Formation Training](https://link.springer.com/chapter/10.1007/978-3-031-56027-9_20)
    - [2024--Explaining Expert Search and Team Formation Systems with ExES](https://arxiv.org/pdf/2405.12881)
    - [2023--A Variational Neural Architecture for Skill-based Team Formation](https://doi.org/10.1145/3589762)
    - [2023--Learning Heterogeneous Subgraph Representations for Team Discovery](https://doi.org/10.21203/rs.3.rs-2318594/v1)
    - [2023--ransfer Learning with Graph Attention Networks for Team Recommendation](https://doi.org/10.1109/ijcnn54540.2023.10191717)
    - [2023--Self-Supervised Primal-Dual Learning for Constrained Optimization](https://ojs.aaai.org/index.php/AAAI/article/view/25520)
    - [2022--Effective Neural Team Formation via Negative Samples](https://doi.org/10.1145/3511808.3557590)
    - [2022--OpeNTF: A Benchmark Library for Neural Team Formation](https://doi.org/10.1145/3511808.3557526)
    - [2022--A Neural Approach to Forming Coherent Teams in Collaboration Networks](https://openproceedings.org/2022/conf/edbt/paper-135.pdf)
    - [2022--Subgraph Representation Learning for Team Mining](https://doi.org/10.1145/3501247.3531578)
    - [2021--Retrieving Skill-Based Teams from Collaboration Networks](https://doi.org/10.1145/3404835.3463105)
    - [2021--Pytfl: A python-based neural team formation toolkit](https://dl.acm.org/doi/pdf/10.1145/3459637.3481992)
    - [2020--Learning to Form Skill-based Teams of Experts](https://doi.org/10.1145/3340531.3412140)
    - [2020--Learning the Value of Teamwork to Form Efficient Teams](https://doi.org/10.1609/aaai.v34i05.6192)
    - [2020--Negotiating team formation using deep reinforcement learning](https://doi.org/10.1016/j.artint.2020.103356)
      <a name="deep-neural-networks-for-optimal-team-composition"></a>
    - [2019--Deep Neural Networks for Optimal Team Composition](https://doi.org/10.3389/fdata.2019.00014)
    - [2019--Optimized group formation for solving collaborative tasks](https://doi.org/10.1007/s00778-018-0516-7)
    - [2013--Weighted synergy graphs for effective team formation with heterogeneous ad hoc agents](https://doi.org/10.1016/j.artint.2013.12.002)
    - [2007--Local strategy learning in networked multi-agent team formation](https://doi.org/10.1007/s10458-006-0007-x)
  - **crowdsourcing**
      - [2022--Crowdsourcing Team Formation With Worker-Centered Modeling](https://doi.org/10.3389/frai.2022.818562)
      - [2021--Self-Organizing Teams in Online Work Settings](https://arxiv.org/pdf/2102.07421)
      - [2021--Understanding Matchmakers’ Experiences, Principles and Practices of Assembling Innovation Teams](https://doi.org/10.1007/s10606-021-09413-4)
      - [2020--What makes a team successful?](https://scholar.google.ca/scholar?q=related:esjtpXM-vHUJ:scholar.google.com/&scioq=What+makes+a+team+successful%3F&hl=en&as_sdt=0,5)      
   - **game theory**
        - [2020--Negotiating team formation using deep reinforcement learning](https://doi.org/10.1016/j.artint.2020.103356)
   - **learning-search-based**
        - [2020--Using machine learning to predict links and improve Steiner tree solutions to team formation problems - a cross company study](https://doi.org/10.1007/s41109-020-00306-x)
   - **reinforcement learning**
        - [2024--Graph Enhanced Reinforcement Learning for Effective Group Formation in Collaborative Problem Solving](https://arxiv.org/pdf/2403.10006)
        - [2020--Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://www.jmlr.org/papers/volume21/20-081/20-081.pdf)
        - [2020--Google Research Football A Novel Reinforcement Learning Environment](https://doi.org/10.1609/aaai.v34i04.5878)
        - [2019--Negotiating team formation using deep reinforcement learning](https://doi.org/10.1016/j.artint.2020.103356)
        - [2018--In Search of the Dream Team: Temporally Constrained Multi-Armed Bandits for Identifying Effective Team Structures](https://doi.org/10.1145/3173574.3173682)
        - [2004--Organization-based cooperative coalition formation](https://doi.org/10.1109/iat.2004.1342939)
- **multi objectives**
   - [2021--A unified framework for effective team formation in social networks](https://doi.org/10.1016/j.eswa.2021.114886)
   - [2018--A multi-objective formulation of the team formation problem in social networks: preliminary results](https://doi.org/10.1145/3205455.3205634)
   - [2018--Effective Team Formation in Expert Networks](https://dblp.org/rec/conf/amw/ZihayatAGKS18.html?view=bibtex)
   - [2017--Authority-Based Team Discovery in Social Networks](https://arxiv.org/pdf/1611.02992)
   - [2014--Two-Phase Pareto Set Discovery for Team Formation in Social Networks](https://ieeexplore.ieee.org/abstract/document/6927639)
   - [2013--Finding Affordable and Collaborative Teams from a Network of Experts](https://doi.org/10.1137/1.9781611972832.65)
   - [2012--Efficient Bi-objective Team Formation in Social Networks](https://doi.org/10.1007/978-3-642-33486-3_31)   
   - [2012--Online Team Formation in Social Networks](https://doi.org/10.1145/2187836.2187950)
   - [2012--Capacitated Team Formation Problem on Social Networks](https://doi.org/10.1145/2339530.2339690)
   - [2011--T-RecS: Team Recommendation System through Expertise and Cohesiveness](https://dl.acm.org/doi/pdf/10.1145/1963192.1963289)
   - [2011--Interaction mining and skill-dependent recommendations for multi-objective team composition](https://doi.org/10.1016/j.datak.2011.06.004)
   - [2010--The community-search problem and how to plan a successful cocktail party](https://doi.org/10.1145/1835804.1835923)
- **number of formed teams**
  - **set of teams**
    - **pareto set**
      - [2014--Two-Phase Pareto Set Discovery for Team Formation in Social Networks](https://ieeexplore.ieee.org/abstract/document/6927639)
      - [2013--Finding Affordable and Collaborative Teams from a Network of Experts](https://doi.org/10.1137/1.9781611972832.65)
    - **top k**
      - [2018--Graph Pattern Matching for Dynamic Team Formation](https://arxiv.org/pdf/1801.01012)
      - [2016--Top-k Team Recommendation in Spatial Crowdsourcing](https://doi.org/10.1007/978-3-319-39937-9_15)
      - [2015--Replacing the Irreplaceable: Fast Algorithms for Team Member Recommendation](https://doi.org/10.1145/2736277.2741132) 
      - [2011--Discovering top-k teams of experts with/without a leader in social networks](https://doi.org/10.1145/2063576.2063718) 
    - **other**
      - [2021--A unified framework for effective team formation in social networks](https://doi.org/10.1016/j.eswa.2021.114886)
      - [2020--Team Recommendation Using Order-Based Fuzzy Integral and NSGA-II in StarCraft](https://doi.org/10.1109/access.2020.2982647)
      - [2020--The sociotechnical teams formation problem: a mathematical optimization approach](https://link.springer.com/content/pdf/10.1007/s10479-018-2759-5.pdf)
  - **single team**
      - [2020--Using machine learning to predict links and improve Steiner tree solutions to team formation problems - a cross company study](https://doi.org/10.1007/s41109-020-00306-x)
      - [2017--Authority-Based Team Discovery in Social Networks](https://arxiv.org/pdf/1611.02992)
      - [2012--TeamFinder: A Co-clustering based Framework for Finding an Effective Team of Experts in Social Networks](https://doi.org/10.1109/icdmw.2012.54)
      - [2012--Efficient Bi-objective Team Formation in Social Networks](https://doi.org/10.1007/978-3-642-33486-3_31)
      - [2012--Capacitated Team Formation Problem on Social Networks](https://doi.org/10.1145/2339530.2339690)  
      - [2011--Discovering top-k teams of experts with/without a leader in social networks](https://doi.org/10.1145/2063576.2063718)
      - [2009--Finding a Team of Experts in Social Networks](https://www.sciencedirect.com/science/article/pii/S2666222121000022)
- **OR-based**
  - **fuzzy**
     - [2020--Team Recommendation Using Order-Based Fuzzy Integral and NSGA-II in StarCraft](https://doi.org/10.1109/access.2020.2982647)
     - [2019--A modified fuzzy approach to project team selection](https://www.sciencedirect.com/science/article/pii/S2666222121000022)
     - [2009--A team formation model based on knowledge and collaboration](https://doi.org/10.1016/j.eswa.2008.12.031)
     - [2007--Project team selection using fuzzy optimization approach](https://doi.org/10.1080/01969720601139041)
  - **genetic algorithm**
     - [2020--Team Recommendation Using Order-Based Fuzzy Integral and NSGA-II in StarCraft](https://doi.org/10.1109/access.2020.2982647)
     - [2019--Application of Genetic Algorithms to the Multiple Team Formation Problem](https://arxiv.org/pdf/1903.03523)
     - [2018--A Multi-objective Formulation of the Team Formation Problem in Social Networks](https://doi.org/10.1145/3205455.3205634)
     - [2014--Non-additive multi-objective robot coalition formation](https://doi.org/10.1016/j.eswa.2013.11.044)
     - [2014--A Method for Group Formation Using Genetic Algorithm](https://www.researchgate.net/profile/Azman-Yasin/publication/229035954_A_Method_for_Group_Formation_Using_Genetic_Algorithm/links/0deec53852db22592c000000/A-Method-for-Group-Formation-Using-Genetic-Algorithm.pdf)
     - [2011--A genetic algorithm approach for group formation in collaborative learning considering multiple student characteristics](https://doi.org/10.1016/j.compedu.2011.09.011)
     - [2009--A team formation model based on knowledge and collaboration](https://doi.org/10.1016/j.eswa.2008.12.031)
  - **graph + OR**
     - [2022--Crowdsourcing Team Formation With Worker-Centered Modeling](https://doi.org/10.3389/frai.2022.818562)
     - [2020--The sociotechnical teams formation problem: a mathematical optimization approach](https://link.springer.com/content/pdf/10.1007/s10479-018-2759-5.pdf)
     - [2019--Optimized group formation for solving collaborative tasks](https://doi.org/10.1007/s00778-018-0516-7)
     - [2018--A Multi-objective Formulation of the Team Formation Problem in Social Networks](https://doi.org/10.1145/3205455.3205634)
     - [2017--TeamGen: An Interactive Team Formation System Based on Professional Social Network](https://doi.org/10.1145/3041021.3054725)
     - [2013--Towards realistic team formation in social networks based on densest subgraphs](https://doi.org/10.1145/2488388.2488482)
     - [2011--Interaction mining and skill-dependent recommendations for multi-objective team composition](https://doi.org/10.1016/j.datak.2011.06.004)
  - **hierarchical**
     - [2017--TeamGen: An Interactive Team Formation System Based on Professional Social Network](https://doi.org/10.1145/3041021.3054725)
     - [2016--Forming Grouped Teams with Efficient Collaboration in Social Networks](https://doi.org/10.1093/comjnl/bxw088)
  - **integer programming**
     - [2024--Leveraging Multicriteria Integer Programming Optimization for Effective Team Formation](https://doi.org/10.1109/tlt.2024.3401734)
     - [2020--The sociotechnical teams formation problem: a mathematical optimization approach](https://link.springer.com/content/pdf/10.1007/s10479-018-2759-5.pdf)
     - [2014--Using hybrid scheduling for the semi-autonomous formation of expert teams](https://doi.org/10.1016/j.future.2013.04.008)
     - [2010--Power in Unity: Forming Teams in Large-Scale Community Systems](https://dl.acm.org/doi/pdf/10.1145/1871437.1871515)
     - [2004--Forming effective worker teams with multi-functional skill requirements](https://doi.org/10.1016/j.cie.2004.12.014)
     - [2001--Forming effective worker teams for cellular manufacturing](https://doi.org/10.1080/00207540110040466) 
  - **linear programming**
     - [2013--Towards realistic team formation in social networks based on densest subgraphs](https://doi.org/10.1145/2488388.2488482)
  - **multi factorial optimization**
     - [2022--An information entropy-based evolutionary computation for multi-factorial optimization](https://doi.org/10.1016/j.asoc.2021.108071)
     - [2022--A Multifactorial Optimization Framework Based on Adaptive Intertask Coordinate System](https://doi.org/10.1109/tcyb.2020.3043509)
     - [2021--An information entropy-based evolutionary computation formulti-factorial optimization](https://www.sciencedirect.com/science/article/pii/S1568494621009765)
     - [2021--Distance Minimization Problems for Multi-factorial Evolutionary Optimization Benchmarking](https://doi.org/10.1007/978-3-030-73050-5_69)
     - [2020--Multi-factorial Optimization for Large-scale Virtual Machine Placement in Cloud Computing](https://arxiv.org/pdf/2001.06585)
     - [2017--Multiobjective Multifactorial Optimization in Evolutionary Multitasking](https://doi.org/10.1109/tcyb.2016.2554622)
     - [2016--Multifactorial Evolution Toward Evolutionary Multitasking](https://doi.org/10.1109/tevc.2015.2458037)
  - **simulation**
     - [2022--Crowdsourcing Team Formation With Worker-Centered Modeling](https://doi.org/10.3389/frai.2022.818562)
  - **task generation**
     - [2024--A Strategy for Implementing Domain-Based Task Generation and Evaluation System Using Text-Text Generative Models](https://link.springer.com/chapter/10.1007/978-981-97-3883-0_3)
     - [2019--A Comparison of Random Task Graph Generation Methods for Scheduling Problems](https://arxiv.org/pdf/1902.05808)
     - [2006--A Task Generation Method for the Development of Embedded Software](https://link.springer.com/chapter/10.1007/11758549_122)
  - **other**
      - [2024--A multi-objective formulation for the team formation problem using Krippendorff’s disagreement and sociometric cohesion with Pareto-solutions obtained via evolutionary algorithms](https://www.sciencedirect.com/science/article/pii/S0305054823003088)
      - [2019--Optimized group formation for solving collaborative](https://doi.org/10.1007/s00778-018-0516-7)
      - [2019--Application of Genetic Algorithms to the Multiple Team Formation Problem](https://arxiv.org/pdf/1903.03523)
      - [2016--Top-k Team Recommendation in Spatial Crowdsourcing](https://doi.org/10.1007/978-3-319-39937-9_15)
      - [2014--Neuro Role and member selection in team formation using resource estimation for large-scale multi-agent systems](https://doi.org/10.1016/j.neucom.2014.04.059)
      - [2011--Interaction mining and skill-dependent recommendations for multi-objective team composition](https://doi.org/10.1016/j.datak.2011.06.004)
      - [2010--Breaking out of the Box of Recommendations: From Items to Packages](https://doi.org/10.1007/s11704-012-2014-1)
      - [2010--Forming reasonably optimal groups: (FROG)](https://doi.org/10.1145/1880071.1880094)
- **similarity between two graphs**
  - **pattern graph**
    - [2018--Graph Pattern Matching for Dynamic Team Formation](https://arxiv.org/pdf/1801.01012)
    - [2020--Efficient Team Formation in Social Networks based on Constrained Pattern Graph](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9101508)
  - **kernel graph**
    - [2015--Replacing the Irreplaceable: Fast Algorithms for Team Member Recommendation](https://doi.org/10.1145/2736277.2741132) 
- **surveys**
  - **team formation**
    - [2021--A Comprehensive Review and a Taxonomy Proposal of Team Formation Problems](https://dl.acm.org/doi/pdf/10.1145/3465399)
    - [2020--A Taxonomy of Team-Assembly Systems: Understanding How People Use Technologies to Form Teams](https://doi.org/10.1145/3415252)
    - [2015--A comparative study of team formation in social networks](https://doi.org/10.1007/978-3-319-18120-2_23)
    - [2000--Teamwork in multi person systems a review and analysis](https://doi.org/10.1080/00140130050084879)
  - **samples**
    - [2025--Personalized Generation In Large Model Era: A Survey](https://arxiv.org/pdf/2503.02614)
    - [2025--Status and Opportunities of Machine Learning Applications in Obstructive Sleep Apnea: A Narrative Review](https://www.sciencedirect.com/science/article/pii/S2001037025001527)
    - [2024--A Comprehensive Survey on Deep Graph Representation Learning](https://doi.org/10.1016/j.neunet.2024.106207)
    - [2024--A Survey on Hypergraph Neural Networks: An In-Depth and Step-by-Step Guide](https://dl.acm.org/doi/pdf/10.1145/3637528.3671457)
    - [2024--Fairness-Aware Graph Neural Networks A Survey](https://doi.org/10.1145/3649142)
    - [2022--Evolutionary design of neural network architectures: a review of three decades of research](https://link.springer.com/content/pdf/10.1007/s10462-021-10049-5.pdf)
    - [2022--A Survey on Deep Learning for Named Entity Recognition](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9039685)
    - [2021--Graph Neural Networks for Natural Language Processing: A Survey](https://www.nowpublishers.com/article/Details/MAL-096)
    - [2021--Metalearning with graph neural networks: Methods and applications](https://dl.acm.org/doi/pdf/10.1145/3510374.3510379)
    - [2021--Affective Image Content Analysis: Two Decades Review and New Perspectives](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9472932)
    - [2020--Team Formation in Software Engineering A Systematic Mapping Study](https://doi.org/10.1109/access.2020.3015017)
    - [2019--A survey on network embedding ](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=8392745)
    - [2019--Deep Learning based Recommender System: A Survey and New Perspectives](https://ieeexplore.ieee.org/abstract/document/9039685)
    - [2018--Inferring User Interests in Microblogging Social Networks: A Survey](https://link.springer.com/content/pdf/10.1007/s11257-018-9207-8.pdf)
    - [2016--USTF A Unified System of Team Formation](https://doi.org/10.1109/tbdata.2016.2546303)
    - [2000--Teamwork in multi person systems a review and analysis](https://www.tandfonline.com/doi/pdf/10.1080/00140130050084879)   
- **team size**
   - **at least k person for each skill**
     - [2012--On team formation with expertise query in collaborative social networks](https://doi.org/10.1007/s10115-013-0695-x)
     - [2012--Multi-skill Collaborative Teams based on Densest Subgraphs](https://doi.org/10.1137/1.9781611972825.15)
     - [2010--Team Formation for Generalized Tasks in Expertise Social Networks](https://doi.org/10.1109/socialcom.2010.12)
   - **at most k responsiblity for each person**
     - [2012--Capacitated Team Formation Problem on Social Networks](https://doi.org/10.1145/2339530.2339690)  
   - **small teams**
     - [2020--Using machine learning to predict links and improve Steiner tree solutions to team formation problems - a cross company study](https://doi.org/10.1007/s41109-020-00306-x)
   - **other**
     - [2020--The sociotechnical teams formation problem: a mathematical optimization approach](https://link.springer.com/content/pdf/10.1007/s10479-018-2759-5.pdf)
     - [2013--Towards realistic team formation in social networks based on densest subgraphs](https://doi.org/10.1145/2488388.2488482)
     - [2010--Power in Unity: Forming Teams in Large-Scale Community Systems](https://dl.acm.org/doi/pdf/10.1145/1871437.1871515)
     - [2010--The community-search problem and how to plan a successful cocktail party](https://doi.org/10.1145/1835804.1835923)
     - [2009--Finding a Team of Experts in Social Networks](https://doi.org/10.1145/1557019.1557074)
- **teams with leader**
  - [2016--Forming Grouped Teams with Efficient Collaboration in Social Networks](https://doi.org/10.1093/comjnl/bxw088)
  - [2011--Discovering top-k teams of experts with/without a leader in social networks](https://doi.org/10.1145/2063576.2063718)
  - [2009--A team formation model based on knowledge and collaboration](https://doi.org/10.1016/j.eswa.2008.12.031)
  

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
[^16]: M. J. Ahmed et al. Vector Representation Learning of Skills for Collaborative Team Recommendation. https://link.springer.com/chapter/10.1007/978-981-96-0567-5_15.
[^17]: H. Fani et al. A Streaming Approach to Neural Team Formation Training. https://link.springer.com/chapter/10.1007/978-3-031-56027-9_20. 
[^18]: R. Barzegar et al. Adaptive Loss-based Curricula for Neural Team Recommendation. https://dl.acm.org/doi/pdf/10.1145/3701551.3703574.
[^19]: K. Thang et al. Translative Neural Team Recommendation: From Multilabel Classification to Sequence Prediction.https://hosseinfani.github.io/res/papers/2025_SIGIR_Translative_Neural_Team_Recommendation_From_Multilabel_Classification_to_Sequence_Prediction.pdf.
[^20]: A. Sapienza et al. {Deep Neural Networks for Optimal Team Composition.https://www.frontiersin.org/journals/big-data/articles/10.3389/fdata.2019.00014/full.
[^21]: R. Hamidi Rad et al. Learning to Form Skill-based Teams of Experts. https://dl.acm.org/doi/pdf/10.1145/3340531.3412140.
[^22]: R. Hamidi Rad et al. A Variational Neural Architecture for Skill-based Team Formation. https://dl.acm.org/doi/pdf/10.1145/3589762.
[^23]: B. Liu et al. Coach-Player Multi-agent Reinforcement Learning for Dynamic Team Composition. https://proceedings.mlr.press/v139/liu21m/liu21m.pdf.









