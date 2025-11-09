import {useEffect} from "react";
import "./Home.css";
import {isDisabled} from "bootstrap/js/src/util";

function Home() {
    let offsets = [];
    useEffect(() => {
        window.addEventListener("load", onWindowLoaded);
        window.addEventListener("scroll", onWindowScroll);
    }, []);

    const onWindowLoaded = () => {
        const ids = ["location","abstract","audience","outline","searchbased","learnbased","refinement","challeng","apps","handson","presenters"];
        const topMargin = 10 //parseInt(getComputedStyle(window.document.body).fontSize) * 10;
        offsets = ids.map((id) => {
            const offset = document.getElementById("section-title-" + id)?.offsetTop - topMargin - 15;
            return {id, offset, endOffset: offset + document.getElementById(`section-${id}`).offsetHeight - 20,};
        });
    };

    const scrollToItem = (e) => {
        const topMargin = 10 //parseInt(getComputedStyle(window.document.body).fontSize) * 10;
        const scroll = document.getElementById(`section-title-${e.target.id}`).offsetTop;
        window.scrollTo({ top: scroll - topMargin,});
    };

    const activeMenuItem = (id) => {
        document.querySelector(".side-menu li.active")?.classList?.remove("active");
        document.getElementById(id).classList.add("active");
    };

    const activeSection = (id) => {
        document.querySelector(".home-content section.active")?.classList?.remove("active");
        document.getElementById(`section-${id}`).classList.add("active");
    };

    const onWindowScroll = (e) => {
        e.preventDefault();
        const currentPosition = window.pageYOffset;
        for (let index = 0; index < offsets.length; index++) {
            const section = offsets[index];
            if (section.offset <= currentPosition && section.endOffset > currentPosition) {
                activeMenuItem(section.id);
                activeSection(section.id);
                break;
            }
        }
    };

    return (
        <div className="home-content">
            <section id="section-location" class="active">
            <span id="section-title-location" className="section-title">Time and Location</span>
                <div className="section-body" style={{textAlign: "center"}}>
                    Half-day, Morning, 9:00 AM - 12:30 PM (GMT/UTC +9)<br/>
                    Monday, November 10, 2025<br/>
                    Room 209A, <a target="_blank" href="https://www.coexcenter.com/">COEX</a>, Gangnam, Seoul, South Korea{" "}
                    <br/><br/>
                    <span style={{ fontSize: '14px', color: 'black' }}>
                        Prev Tutorials at <a target="_blank" href="https://fani-lab.github.io/OpeNTF/tutorial/umap24/" >UMAP24</a>, <a target="_blank" href="https://fani-lab.github.io/OpeNTF/tutorial/sigir-ap24/" >SIGIR-AP24</a> and <a target="_blank" href="https://fani-lab.github.io/OpeNTF/tutorial/wsdm25/" >WSDM25</a> ◁ [<a target="_blank" href="https://hosseinfani.github.io/res/papers/2025_CIKM_Neural_Shifts_in_Collaborative_Team_Recommendation.pdf">Full Outline</a>]
                        [<a target="_blank" style={{color: "gray"}}>Slides</a>]
                        [<a target="_blank" style={{color: "gray"}}>Recording</a>]
                    </span>
                </div>
            </section>
            <section id="section-abstract"><span id="section-title-abstract" className="section-title">Abstract</span>
                    <div className="section-body justify-paragraph"style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                      <img src={require("../img/venn.png")} alt="venn." height="350" />
                      <p> Figure 1: Overlapping Team Formation Approaches.</p>
                    </div>

                <div className="section-body justify-paragraph">
                    Team recommendation involves selecting skilled experts to form
                    an almost surely successful collaborative team, or refining the team
                    composition to maintain or excel at performance. To eschew the
                    tedious and error-prone manual process, various computational (Figure 1) and
                    social science theoretical approaches have been proposed wherein
                    the problem definition remains essentially the same, while it has
                    been referred to by such other names as team allocation, selection, composition, and formation. In this tutorial, we study the
                    advancement of computational approaches from greedy search in
                    pioneering works to the recent learning-based approaches, with
                    a particular in-depth exploration of graph neural network-based
                    methods as the cutting-edge class, via unifying definitions, formulations, and evaluation schema. More importantly, we then discuss
                    team refinement, a subproblem in team recommendation that involves structural adjustments or expert replacements to enhance
                    team performance in dynamic environments. Finally, we introduce
                    training strategies, benchmarking datasets, and open-source tools,
                    along with future research directions and real-world applications.
                    </div>
                    <div className="section-body justify-paragraph"style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                      <img src={require("../img/outline2.png")} alt="Tutorial Agenda." height="450" />
                      <p> Figure 2: Tutorial agenda.</p>
                    </div>
            </section>

            <section id="section-audience"><span id="section-title-audience" className="section-title">Target Audience</span>
                <div className="section-body justify-paragraph">
                    Team recommendation falls under social information retrieval
                    (Social IR) where we seek to find the right group of skillful experts to
                    solve the tasks at hand or only with the assistance of social
                    resources. In this tutorial,
                    <ul>
                        <li className="justify-paragraph">
                            We target beginner or intermediate researchers, industry
                            technologists and practitioners with a broad interest in user
                            modeling and recommender systems who are willing to have a whole
                            picture of team recommendation techniques.
                        </li>
                        <li className="justify-paragraph">
                            Furthermore, this tutorial targets audiences from the graph neural
                            network (GNN) community for a comprehensive review of subgraph
                            optimization objectives and calls them for further development of
                            effective yet efficient graph neural networks with a special focus
                            on team recommendation.
                        </li>
                        <li className="justify-paragraph">
                            Last, having regard to the unified comparative analysis, this
                            tutorial enables organizations and practitioners to compare
                            different models and pick the most suitable one for their
                            application to form collaborative teams of experts whose
                            success is almost surely guaranteed.
                        </li>
                    </ul>
                    <div className="justify-paragraph">
                        The target audience needs to be familiar with graph theory and machine
                        learning. Where appropriate, the tutorial will not make any
                        assumptions about the audience’s knowledge on more advanced
                        techniques. As such, sufficient details will be provided as
                        appropriate so that the content will be accessible and understandable
                        to those with a fundamental understanding of such principles.
                    </div>
                </div>
            </section>

            <section id="section-outline"><span id="section-title-outline" className="section-title">Outline</span>
                <div className="section-body justify-paragraph"style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                      <img src={require("../img/taxonomy.png")} alt="Taxonomy of team recommendation methods." height="400" />
                      <p> Figure 3: Taxonomy of team recommendation approaches.</p>
                </div>
                <div className="section-body">
                  <span className="d-block w-100 justify-paragraph">
                    As seen in Figure 2, We start our tutorial with a brief introduction to the pioneering graph-based
                      team recommendation algorithms based on a taxonomy of computational methods, as shown in Figure 3,4,
                      then continue to explore the learning-based team recommendation and team refinement methods, focusing
                      on modern methods based on graph neural networks and reinforcement learning.
                  </span>
                </div>
                <div className="section-body justify-paragraph"style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                      <img src={require("../img/alluvial.png")} alt="Team recommendation methods in time." height="500" />
                      <p> Figure 4: Team recommendation methods in time.</p>
                </div>
            </section>

            <section id="section-searchbased"><span id="section-title-searchbased" className="section-title">Pioneering Techniques </span>
                <div className="section-body justify-paragraph">
                       <span className="d-block w-100 justify-paragraph">
                           The early computational models for team
                           recommendation were developed in operations research (OR), optimizing objectives
                           using integer linear and/or nonlinear programming (IP). Such work, however, was
                           premised on the mutually independent selection of experts and overlooked the
                           organizational and/or social ties. To bridge the gap, graph-based approaches have
                           been proposed to recommend teams via subgraph optimization where the different aspects
                           of real-world teams are captured like communication cost and geographical proximity. This
                           section of our tutorial provides an overview of the graph-based approaches in
                          team recommendation methods.

                       </span>
                       <div className="topic-item">
                            <ul><li className="justify-paragraph">
                                <span className="fw-bold">Subgraph Optimization Objectives:</span>
                                    &nbsp;In our tutorial, we formalized more than 13 objectives
                                    in a unified framework with integrated notations for better
                                    readability and fostering conventions in this realm.
                                </li>
                                <li className="justify-paragraph">
                                  <span className="fw-bold">Subgraph Optimization Techniques:</span>
                                    &nbsp;We describe the seminal heuristics that have been
                                    followed by the majority of researchers, as well as the groups
                                    that optimization techniques can be studied in.
                                </li>
                                <li className="justify-paragraph">
                                    <span className="fw-bold">Evaluation Methodology:</span>
                                    &nbsp;Finally, we lay out the methodologies, benchmark
                                    datasets, and quantitative and qualitative metrics that are
                                    used to evaluate the performance of the graph-based
                                    approaches.
                                </li>
                            </ul>
                       </div>



                <div className="section-topic justify-paragraph"style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>

                               <img src={require("../img/optimization.jpg")} alt="Subgraph optimization using different objectives for team recommendation problem." height="200"/>
                               <p> Figure 5 . Subgraph optimization using different objectives for team recommendation problem.</p>
                </div>

                <div className="topic-item">
                            <span className="fw-bold text-uppercase h6">Reading List</span>
                            <ul>
                                <li><a target="_blank" href="https://dl.acm.org/doi/full/10.1145/3737455">A Survey of subgraph optimization for expert team formation</a>{" "}(Saeedi et al., 2025)</li>
                                <li><a target="_blank" href="https://doi.org/10.1016/j.eswa.2021.114886">A unified framework for effective team formation in social networks</a>{" "}(Selvarajah et al., 2021)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/3459637.3481969">RW-Team: Robust team formation using random walk</a>{" "}(Nemec et al., 2021)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/3465399">A Comprehensive review and a taxonomy proposal of team formation Problems</a>{" "}(Juarez et al., 2021)</li>
                                <li><a target="_blank" href="https://doi.org/10.1109/TKDE.2020.2985376">Effective keyword search over weighted graphs</a>{" "}(Kargar et al., 2020)</li>
                                <li><a target="_blank" href="https://doi.org/10.1093/comjnl/bxw088">Forming grouped teams with efficient collaboration in social networks</a>{" "}(Huang et al., 2017)</li>
                                <li><a target="_blank" href="https://doi.org/10.5441/002/edbt.2017.54">Authority-based team discovery in social networks</a>{" "}(Zihayat et al., 2017)</li>
                                <li><a target="_blank" href="https://doi.org/10.1109/WI-IAT.2014.112">Two-phase pareto set discovery for team formation in social networks</a>{" "}(Zihayat et al., 2014)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/2488388.2488482">Towards realistic team formation in social networks based on densest subgraphs</a>{" "}(Rangapuram et al., 2013)</li>
                                <li><a target="_blank" href="https://doi.org/10.1137/1.9781611972825.15">Multi-skill collaborative teams based on densest subgraphs</a>{" "}(Gajewar et al., 2012)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/2339530.2339690">Capacitated team formation problem on social networks</a>{" "}(Datta et al., 2012)</li>
                                <li><a target="_blank" href="https://doi.org/10.1109/ICDMW.2011.28">An effective expert team formation in social networks based on skill grading</a>{" "}(Farhadi et al., 2011)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/2063576.2063718">Discovering top-k teams of experts with/without a leader in social networks</a>{" "}(Kargar et al., 2011)</li>
                                <li><a target="_blank" href="https://doi.org/10.1109/SocialCom.2010.12">Team formation for generalized tasks in expertise social networks</a>{" "}(Li et al., 2010)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/1557019.1557074">Finding a team of experts in social networks</a>{" "}(Lappas et al., 2009)</li>
                            </ul>
                </div>
                </div>
            </section>

            <section id="section-learnbased"><span id="section-title-learnbased" className="section-title">Neural Team Recommendation </span>
                    <div className="section-body">
                        <span className="d-block w-100 justify-paragraph">
                          We will then explain the learning-based methods, which has been
                          mostly based on neural models. Learning-based methods bring
                          efficiency while enhancing efficacy due to the inherently
                          iterative and online learning procedure, and can address the
                          limitations of search-based solutions with respect to scalability,
                          as well as dynamic expert networks (<a target="_blank" href="https://dl.acm.org/doi/10.1145/3589762">Rad, R. et al., 2023</a>;{" "}<a target="_blank" href="https://dl.acm.org/doi/10.1145/3340531.3412140">Rad, R.H., et al., 2020</a>). We will lay out the details of different neural
                                    architecture and their applications in team recommendation,
                                    from autoencoder to graph neural networks.
                        </span>
                        <div className="topic-item">
                            <ul>

                                <div className="section-topic justify-paragraph"style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                                    <img src={require("../img/e2e.png")} alt="STop: Graph representation learning of skills." height="300"/>
                                    <p>Figure 6. End-to-End Graph neural team recommendation </p>
                                </div>
                                <div className="section-topic justify-paragraph"style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                                    <img src={require("../img/s2s.png")} alt="seq-to-seq." height="300"/>
                                    <p>Figure 7. : Multilabel vs. seq-to-seq team recommendation.</p>
                                </div>
                                <li className="justify-paragraph">
                                    <span className="fw-bold">Training Strategies:</span>
                                    &nbsp;In our tutorial, we will discuss the details of
                                    different negative sampling heuristics to draw virtually
                                    unsuccessful teams and streaming training strategy that put a
                                    chronological order on teams during training.
                                </li>
                                <div className="section-topic justify-paragraph"style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                                    <img src={require("../img/flow-t.jpg")} alt="Streaming training strategy in neural-based team formation methods." height="350" />
                                    <p>Figure 8. Streaming training strategy in neural-based team formation methods. </p>
                                </div>

                                 </ul>
                        </div>
                        <div className="topic-item">
                            <span className="fw-bold text-uppercase h6">Reading List</span>
                            <ul>
                                <li><a target="_blank" href="https://link.springer.com/chapter/10.1007/978-981-96-0567-5_15">Vector representation learning of skills for collaborative team recommendation: a comparative study</a> (Ahmed et al., 2025)</li>
                                <li><a target="_blank" href="https://link.springer.com/chapter/10.1007/978-3-031-56027-9_20">A streaming approach to neural team formation training</a> (Fani et al., 2024)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/3589762">A variational neural architecture for skill-based team formation</a>{" "}(Rad et al., 2023)</li>
                                <li><a target="_blank" href="https://doi.org/10.1109/IJCNN54540.2023.10191717">Transfer learning with graph attention networks for team recommendation</a>{" "}(Kaw et al., 2023)</li>
                                <li><a target="_blank" href="https://doi.org/10.1007/s10791-023-09421-6">Learning heterogeneous subgraph representations for team discovery</a> (Nguyen et al., 2023)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/3511808.3557526">OpeNTF: A benchmark library for neural team formation</a> (Dashti et al., 2022)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/3511808.3557590">Effective neural team formation via negative samples</a> (Dashti et al., 2022)</li>
                                <li><a target="_blank" href="https://doi.org/10.48786/edbt.2022.37">A neural approach to forming coherent teams in collaboration networks</a> (Rad et al., 2022)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/3501247.3531578">Subgraph representation learning for team mining</a> (Rad et al., 2022)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/3404835.3463105">Retrieving skill-based teams from collaboration networks</a> (Rad et al., 2021)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/3459637.3481992">PyTFL: A python-based neural team formation toolkit</a> (Rad et al., 2021)</li>
                                <li><a target="_blank" href="https://doi.org/10.3389%2Ffdata.2019.00014">Deep neural networks for optimal team composition</a> (Sapienza et al., 2019)</li>
                                <li><a target="_blank" href="https://dl.acm.org/doi/10.1145/3340531.3412140">Learning to form skill-based teams of experts</a> (Rad et al., 2019)</li>
                                <li><a target="_blank" href="https://dl.acm.org/doi/10.1145/3097983.3098036">metapath2vec: Scalable representation learning for heterogeneous networks</a> (Dong et al., 2017)</li>
                                <li><a target="_blank" href="https://www.semanticscholar.org/paper/Representation-Learning-on-Graphs%3A-Methods-and-Hamilton-Ying/ecf6c42d84351f34e1625a6a2e4cc6526da45c74">Representation learning on graphs: methods and applications</a> (Hamilton et al., 2017)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/1935826.1935914">Supervised random walks: predicting and recommending links in social networks</a> (Backstrom et al., 2011)</li>
                            </ul>
                        </div>
                        <div className="topic-item">
                            <span className="fw-bold expand-button"></span>
                            <span className="d-block w-100 justify-paragraph"></span>
                        </div>
                    </div>
            </section>

                    <section id="section-refinement"><span id="section-title-refinement" className="section-title">Team Refinement </span>
                <div className="section-body justify-paragraph">
                        <div className="topic-item">
                            Reinforcement learning with neural policy estimators has been increasingly employed to learn the dynamics of real-world teams
                            for structural modifications or team member replacements in order to maintain or even improve team effectiveness by
                            modeling the problem in a multi-agent setting where a group of agents synchronize their actions in a decentralized manner within
                            a shared environment to achieve a common goal. The task cannot be completed by any individual agent alone but in a team of agents,
                            and each agent must make a decision regarding which team to join.
                        </div>

                        <div className="section-topic justify-paragraph"style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                      <img src={require("../img/refinement.png")} alt="team refinement." height="400" />
                      <p> Figure 9. Expert replacement as a team refinement method (<a target="_blank" href="https://www.sciencedirect.com/science/article/pii/S0167923624001763">Lv et al., 2024</a>).</p>
                        </div>
                        <div className="topic-item">
                            <span className="fw-bold text-uppercase h6">Reading List</span>
                            <ul>
                                <li><a target="_blank" href="https://dl.acm.org/doi/pdf/10.1145/3729176.3729197">taifa: Enhancing team effectiveness and cohesion with ai-generated automated feedback.</a> (Almutairi et al., 2025)</li>
                                <li><a target="_blank" href="https://www.sciencedirect.com/science/article/pii/S0167923624001763">Team formation in large organizations: A deep reinforcement learning approach.</a> (Lv et al., 2024)</li>
                                <li><a target="_blank" href="https://proceedings.neurips.cc/paper_files/paper/2023/file/906c860f1b7515a8ffec02dcdac74048-Paper-Conference.pdf">Automatic grouping for efficient cooperative multi-agent reinforcement learning.</a> (Zang et al., 2023)</li>
                                <li><a target="_blank" href="https://proceedings.mlr.press/v139/liu21m/liu21m.pdf">Coach-player multi-agent reinforcement learning for dynamic team composition.</a> (Liu et al., 2021)</li>
                                <li><a target="_blank" href="https://dl.acm.org/doi/pdf/10.1145/3415252">A taxonomy of team-assembly systems: Understanding how people use technologies to form teams.</a> (Zara et al., 2020)</li>
                                <li><a target="_blank" href="https://www.jmlr.org/papers/v21/20-081.html">Monotonic value function factorisation for deep multi-agent reinforcement learning.</a> (Rashid et al., 2020)</li>
                                <li><a target="_blank" href="https://ojs.aaai.org/index.php/AAAI/article/view/5878">Google research football: A novel reinforcement learning environment.</a> (Kurach et al., 2020)</li>
                                <li><a target="_blank" href="https://www.sciencedirect.com/science/article/pii/S0004370220301077"> Negotiating team formation using deep reinforcement learning.</a> (Bachrach et al., 2020)</li>
                                <li><a target="_blank" href="https://dl.acm.org/doi/pdf/10.1145/3173574.3173682">In search of the dream team: Temporally constrained multi-armed bandits for identifying effective team structures.</a> (Zhou et al., 2018)</li>
                                <li><a target="_blank" href="https://par.nsf.gov/servlets/purl/10062454">Team expansion in collaborative environments.</a> (Zhao et al., 2018)</li>
                                <li><a target="_blank" href="https://dl.acm.org/doi/pdf/10.1145/3134686">Two sides to every story: Mitigating intercultural conflict through automated feedback and shared self-reflections in global virtual teams.</a> (He et al., 2017)</li>
                            </ul>
                        </div>
                    </div>
            </section>

            <section id="section-challeng"><span id="section-title-challeng" className="section-title">Challenges and New Perspectives </span>
                <div className="section-body">
                    <div className="topic-item">
                            <span className="fw-bold">Fair and Diverse Team Recommendation</span>
                            <span className="d-block w-100 justify-paragraph">
                                The primary focus of existing team recommendation methods is the
                                maximization of the success rate for the recommended teams,
                                largely ignoring diversity in the recommended users. In our
                                tutorial, we introduce notions of fairness and in-process, e.g.,
                                <a target="_blank" href="https://github.com/fani-lab/OpeNTF/tree/vivaFemme-bias24"> (vivaFemme)</a> and
                                <a target="_blank" href="https://github.com/fani-lab/OpeNTF/tree/cl-wsdm25"> debiasing learning curricula</a>,
                                and post-processing reranking
                                <a target="_blank" href="https://github.com/fani-lab/Adila"> (Adila) </a> refinements to
                                reassure the desired fair outcome and explore the synergistic
                                trade-offs between notions of fairness and success rate for the
                                proposed solutions.
                              </span>



                        <div className="section-topic justify-paragraph"style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                            <img src={require("../img/vivafemme.jpg")} alt="vivaFemme's loss regularization to mitigate gender bias in team recommendation." height="400" />
                            <p>Figure 10. vivaFemme's loss regularization to mitigate gender bias in team recommendation. [<a target="_blank" href="https://github.com/fani-lab/OpeNTF/tree/vivaFemme-bias24">code</a>][<a target="_blank" href="https://hosseinfani.github.io/res/papers/2024_BIAS_SIGIR_vivaFemme_Mitigating_Gender_Bias_in_Neural_Team_Recommendation_via_Female-Advocate_Loss_Regularization.pdf">paper</a>]</p>
                        </div>

                        <div className="section-topic justify-paragraph"style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                            <img src={require("../img/cl.png")} alt="Adaptive loss-based curricula to mitigate popularity bias." height="200" />
                            <p>Figure 11. Adaptive loss-based curricula to mitigate popularity bias.[<a target="_blank" href="https://github.com/fani-lab/OpeNTF/tree/cl-wsdm25">code</a>][<a target="_blank" href="https://hosseinfani.github.io/res/papers/2025_WSDM_Adaptive_Loss-based_Curricula_for_Neural_Team_Recommendation.pdf">paper</a>]</p>
                        </div>

                        <div className="section-topic justify-paragraph"style={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                            <img src={require("../img/adila.png")} alt="Adila's reranking pipeline." height="300" />
                            <p>Figure 12. Adila's reranking pipeline. [<a target="_blank" href="https://github.com/fani-lab/Adila/tree/bias23">code</a>][<a target="_blank" href="https://hosseinfani.github.io/res/papers/2023_BIAS_ECIR_Bootless_Application_of_Greedy_Re-ranking_Algorithms_in_Fair_Neural_Team_Formation.pdf">paper</a>]</p>
                        </div>

                    </div>
                        <div className="topic-item">
                            <span className="fw-bold">Spatial Team Recommendation</span>
                            <span className="d-block w-100 justify-paragraph">
                                In search of an optimal team, companies further look for skilled
                                users in a region where the organization is geographically
                                based, which leads to new challenges as it requires drilling
                                down on the skills of users while maintaining the condition of a
                                given geolocation. We conclude our tutorial by bringing forth
                                the spatial team recommendation problem; that is, given a set of
                                users, skills and geolocations, the goal is to determine whether
                                the combination of skills and geolocations in forming teams has
                                synergistic effects.
                              </span>
                        </div>
                        <div className="topic-item">
                            <span className="fw-bold">Multi-Objective Optimization</span>
                            <span className="d-block w-100 justify-paragraph">
                                In real-world team recommendation scenarios, balancing multiple, often
                                conflicting objectives (e.g., team effectiveness and experts' workload distribution)
                                requires a training process guided by a loss function that explicitly accounts for
                                multiple objectives. However, existing neural team recommendation approaches, which
                                commonly frame the problem as a classification or link prediction task, aim to maximize
                                the coverage of the required skills and mainly rely on standard loss functions such as
                                cross-entropy, and designing a task-aware loss function is overlooked.
                              </span>
                        </div>
                        <div className="topic-item">
                            <span className="fw-bold text-uppercase h6">Reading List</span>
                            <ul>
                                <li><a target="_blank" href="https://doi.org/10.1145/3701551.3703574">Adaptive loss-based curricula for neural team recommendation.</a> (Barzegar et al., 2025)</li>
                                <li><a target="_blank" href="https://doi.org/10.1007/978-3-031-71975-2_6">vivafemme: Mitigating gender bias in neural team recommendation via female-advocate loss regularization.</a> (Moasses et al., 2024)</li>
                                <li><a target="_blank" href="https://doi.org/10.1007/978-3-031-37249-0_9">Bootless application of greedy re-ranking algorithms in fair neural team formation</a> (Loghmani et al., 2023)</li>
                                <li><a target="_blank" href="https://doi.org/10.1016/j.ipm.2021.102707">Fair top-k ranking with multiple protected groups</a>{" "}(Zehlike et al., 2022)</li>
                                <li><a target="_blank" href="https://doi.org/10.1609/aaai.v36i11.21445">Has ceo gender bias really been fixed? adversarial attacking and improving gender fairness in image search</a> (Feng et al., 2022)</li>
                                <li><a target="_blank" href="http://papers.nips.cc/paper_files/paper/2022/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html">Fair ranking with noisy protected attributes</a> (Mehrotra et al., 2022)</li>
                                <li><a target="_blank" href="https://link.springer.com/article/10.1007/s42452-020-2801-5">Challenges and barriers in virtual teams: a literature review</a> (Morrison et al., 2020)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/3292500.3330691">Fairness-aware ranking in search & recommendation systems with application to linkedin talent search</a> (Geyik et al., 2019)</li>
                                <li><a target="_blank" href="https://doi.org/10.1145/1460563.1460633">Who collaborates successfully? prior experience reduces collaboration barriers in distributed interdisciplinary research</a> (Cummings et al., 2008)</li>
                            </ul>
                        </div>
                </div>
            </section>

                        <section id="section-apps"><span id="section-title-apps" className="section-title">Applications </span>
                <div className="section-body">
                        <div className="topic-item">
                            In this section, we highlight the practical significance of team recommendation by
                            explaining its seemingly unrelated yet highly valuable applications in education,
                            research, and healthcare, in addition to its common use cases.
                            <span className="d-block w-100 justify-paragraph">
                                <ul>
                                  <li><b> Group Learning: </b>
                                    Team recommendation finds immediate application in group-based learning environments.
                                    In online classes, where physical presence and interaction are absent, team recommendation connects students to share ideas and build relationships.
                                    This not only enhances their social skills but also combats the isolation that can sometimes accompany remote learning.
                                    Via working in teams, students are exposed to varying viewpoints, backgrounds, and problem-solving approaches.
                                  </li>
                                  <li><b> Reviewer Assignment: </b>
                                    Another immediate application of team recommendation is in peer-review assignments where a group of reviewers
                                    are paired with manuscripts within the reviewers' expertise for high-quality reviews while managing conflicts of interests.
                                    Like team recommendation, herein, research topics (skills) and reviewers (experts) are mapped into a
                                    latent space and, given a manuscript about a subset of research topics, team recommendation aims to recommend
                                    reviewers with top-k closest vectors to the vectors of the research.
                                  </li>
                                  <li><b> Palliative Care: </b>
                                    Another application of team recommendation is in healthcare, which assigns a team of caregivers to patients who
                                    seek help for their daily activities due to disease or disorders. The challenge lies in optimally assigning care
                                    providers in teams to address patients' needs while considering factors such as communication, distance, and contract costs.
                                  </li>
                                </ul>
                            </span>
                        </div>
                    <div className="topic-item">
                            <span className="fw-bold text-uppercase h6">Reading List</span>
                            <ul>
                                <li><a target="_blank" href="https://ieeexplore.ieee.org/abstract/document/10531676">Leveraging Multicriteria Integer programming optimization for effective team formation.</a> (Singh et al., 2025)</li>
                                <li><a target="_blank" href="https://www.sciencedirect.com/science/article/pii/S0957417424028380">Peer review expert group recommendation: A multi-subject coverage-based approach.</a> (Fu et al., 2025)</li>
                                <li><a target="_blank" href="https://www.sciencedirect.com/science/article/pii/S0957417425007298">Support group formation for users with depression in social networks.</a> (Yang et al., 2025)</li>
                                <li><a target="_blank" href="https://patents.google.com/patent/US20240005230A1/en">A system and method for recommending expert reviewers for performing quality assessment of an electronic work.</a> (Cuzzola et al., 2024)</li>
                                <li><a target="_blank" href="https://dl.acm.org/doi/abs/10.1145/3627673.3679081">Reviewerly: modeling the reviewer assignment task as an information retrieval problem.</a> (Arabzadeh et al., 2024)</li>
                                <li><a target="_blank" href="https://www.emerald.com/bpmj/article/30/1/63/1221421/Understanding-knowledge-leadership-in-improving">Understanding knowledge leadership in improving team outcomes in the health sector:  a Covid-19 study.</a> (Khan et al., 2024)</li>
                                <li><a target="_blank" href="https://link.springer.com/chapter/10.1007/978-3-031-40725-3_34">An integer linear programming model for team formation in the classroom with constraints.</a> (Candel et al., 2023)</li>
                                <li><a target="_blank" href="https://www.jair.org/index.php/jair/article/view/14318">Reviewer assignment problem: A systematic review of the literature.</a> (Aksoy et al., 2023)</li>
                                <li><a target="_blank" href="https://proceedings.neurips.cc/paper_files/paper/2023/hash/ccba10dd4e80e7276054222bb95d467c-Abstract-Conference.html">Group fairness in peer review.</a> (Aziz et al., 2023)</li>
                                <li><a target="_blank" href="https://ieeexplore.ieee.org/abstract/document/10139187">Reviewer assignment decision support in an academic journal based on multicriteria assessment and text mining.</a> (Latypova et al., 2023)</li>
                                <li><a target="_blank" href="https://journals.lww.com/hcmrjournal/fulltext/2021/10000/Hacking_teamwork_in_health_care__Addressing.9.aspx">Hacking teamwork in health care: Addressing adverse effects of ad hoc team composition in critical care medicine.</a> (McLeod et al., 2021)</li>
                                <li><a target="_blank" href="https://dl.acm.org/doi/abs/10.1145/3487664.3487673">Revaside: assignment of suitable reviewer sets for publications from fixed candidate pools.</a> (Kreutz et al., 2021)</li>
                                <li><a target="_blank" href="https://pmc.ncbi.nlm.nih.gov/articles/PMC6361117/pdf/nihms-990788.pdf">Teamwork in healthcare: Key discoveries enabling safer, high-quality care.</a> (Rosen et al., 2018)</li>
                                <li><a target="_blank" href="https://onlinelibrary.wiley.com/doi/full/10.1111/exsy.12038">Predicting teamwork results from social network analysis.</a> (Crespo et al., 2015)</li>
                                <li><a target="_blank" href="https://www.researchgate.net/profile/Azman-Yasin/publication/229035954_A_Method_for_Group_Formation_Using_Genetic_Algorithm/links/0deec53852db22592c000000/A-Method-for-Group-Formation-Using-Genetic-Algorithm.pdf">A method for group formation using genetic algorithm.</a> (Ani et al., 2010)</li>

                            </ul>
                        </div>
                </div>
            </section>

            <section id="section-handson">

                        <span id="section-title-handson" className="section-title">Hands-on OpeNTF </span>
                           <div className="section-body justify-paragraph">
                                    In our tutorial, we introduce publicly available
                                    libraries and tools for the task of team recommendation.
                                    Notably, we provide hands-on experience with{" "}
                                    <a target="_blank" href="https://github.com/fani-lab/OpeNTF"><span className="fw-bold">
                                    {" "}<a target="_blank" href="https://github.com/fani-lab/OpeNTF">OpeNTF</a>
                                  </span></a>
                                    , an open-source benchmark library for neural models that can efficiently preprocess
                               large-scale datasets, can be easily extended or customized to new neural methods, and is extensible to new datasets from other domains.
                           </div>

            </section>



            <section id="section-presenters">
                <span id="section-title-presenters" className="section-title">Presenters</span>
                <div className="section-body d-flex p-3 direction-row justify-content-between">
                    <div className="presenter">
                        <a target="_blank" href="https://www.linkedin.com/in/mahdis-saeedi-b80b8321a/" target="_blank">
                            <div class="presenter-img">
                                <img class="front" src={require("../img/mahdis.jpg")} alt="Mahdis Saeedi"/>
                                <img class="back" src={require("../img/flamingo.jpg")} alt="Flamingo"/>
                            </div>
                        </a>
                        <span className="ref-name fs-5"><a target="_blank" href="https://www.linkedin.com/in/mahdis-saeedi-b80b8321a/" target="_blank">Mahdis Saeedi</a></span>
                         <span className="text-muted fs-10">Postdoctoral Fellow, School of Computer Science<br/>Lecturer, Department of Mathematics<br/> University of Windsor</span>

                    </div>
                    <div className="presenter">
                        <a target="_blank" href="http://hosseinfani.github.io/" target="_blank">
                            <div class="presenter-img">
                                <img class="front" src={require("../img/hossein.jpg")} alt="Hossein Fani"/>
                                <img class="back" src={require("../img/sloth.jpg")} alt="Sloth"/>
                            </div>
                        </a>
                        <span className="ref-name fs-5"><a target="_blank" href="http://hosseinfani.github.io/" target="_blank">Hossein Fani</a></span>
                        <span className="text-muted fs-10">Assistant Professor<br/>School of Computer Science<br/>University of Windsor</span>

                    </div>
                </div>
            </section>

        </div>
    );
}

export default Home;
