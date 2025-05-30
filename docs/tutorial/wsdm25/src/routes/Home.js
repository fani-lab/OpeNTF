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
        const ids = ["location","abstract","audience","outline", "searchbased", "learnbased", "challeng", "apps", "presenters"];
        const topMargin = parseInt(getComputedStyle(window.document.body).fontSize) * 10;
        offsets = ids.map((id) => {
            const offset =
                document.getElementById("section-title-" + id)?.offsetTop - topMargin - 15;
            return {id, offset, endOffset: offset + document.getElementById(`section-${id}`).offsetHeight - 20,};
        });
    };

    const scrollToItem = (e) => {
        const topMargin = parseInt(getComputedStyle(window.document.body).fontSize) * 6;
        const scroll = document.getElementById(`footnote-${e.target.id}`).offsetTop;
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
                    Half-day, Afternoon, 1:30 PM - 5:00 PM (GMT+1)<br/>
                    Monday, March 10, 2025<br/>
                    Konferenzraum 16, Hannover Congress Centre (HCC), Hannover, Germany{" "}
                    <br/><br/>
                    [<a target="_blank" href="https://hosseinfani.github.io/res/papers/2025_WSDM_Bridging_Historical_Subgraph_Optimization_and_Modern_Graph_Neural_Network_Approaches_in_Team_Recommendations.pdf">Full Outline</a>]
                    [<a target="_blank" href="https://hosseinfani.github.io/res/slides/2025_WSDM_Bridging_Historical_Subgraph_Optimization_and_Modern_Graph_Neural_Network_Approaches_in_Team_Recommendations.pdf">Slides</a>]
                    [<a target="_blank" style={{color: "gray"}}>Recording</a>]
                </div>
            </section>
            <section id="section-abstract"><span id="section-title-abstract" className="section-title">Abstract</span>
                <div className="section-body justify-paragraph">
                    Collaborative team recommendation involves selecting experts with
                    certain skills to form a team who will, more likely than not,
                    accomplish a complex task. To automate the traditionally
                    tedious and error-prone manual process of team formation, researchers have proposed methods to tackle the
                    problem. In this tutorial, while providing a taxonomy of team
                    recommendation works based on their algorithmic approaches, we perform a comprehensive and
                    hands-on study of the graph-based and learning-based approaches that comprise the
                    mainstream in this field, then cover the graph neural team recommenders as
                    the cutting-edge class of approaches. Further, we introduce
                    details of training strategies, benchmarking datasets, and open-source
                    tools, along with directions for future works.
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
                <div className="section-body">
                  <span className="d-block w-100 justify-paragraph">
                    As seen in Figure 1, we begin to introduce intuitive definitions
                    of a team and some representative, historical to modern and
                    state-of-the-art methods for solving the team recommendation
                    problem, motivating the importance of the problem, followed by a
                    novel taxonomy of computational methods, as explained hereafter.
                  </span>
                    <img src={require("../img/taxonomy.png")} alt="Taxonomy of team recommendation methods." height="300"/>
                    <img src={require("../img/alluvial.png")} alt="Team recommendation methods in time." height="330"/>
                    <p> Figure 1. Taxonomy of team recommendation methods within time.</p>
                </div>
            </section>
            <section id="section-searchbased"><span id="section-title-searchbased" className="section-title">Search-based Heuristics (35 Minutes)</span>
                    <div className="section-body">
                        <span className="d-block w-100 justify-paragraph">
                          This section provides an overview of the graph-based approaches in
                          team formation methods. Operations Research-based methods,
                          although conceiving the foremost computational models, overlooked
                          the organizational and social ties among users and are hence
                          excluded in our tutorial.
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
                        <div className="topic-item">
                            <span className="fw-bold text-uppercase h6">Reading List</span>
                            <ul>
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
            <section id="section-learnbased"><span id="section-title-learnbased" className="section-title">Learning-based Heuristics (55 Minutes)</span>
                    <div className="section-body">
                        <span className="d-block w-100 justify-paragraph">
                          We will then explain the learning-based methods, which has been
                          mostly based on neural models. Learning-based methods bring
                          efficiency while enhancing efficacy due to the inherently
                          iterative and online learning procedure, and can address the
                          limitations of search-based solutions with respect to scalability,
                          as well as dynamic expert networks (<a target="_blank" href="https://dl.acm.org/doi/10.1145/3589762">Rad, R. et al., 2023</a>;{" "}<a target="_blank" href="https://dl.acm.org/doi/10.1145/3340531.3412140">Rad, R.H., et al., 2020</a>).
                        </span>
                        <div className="topic-item">
                            <ul>
                                <li className="justify-paragraph">
                                    <span className="fw-bold">Neural Architectures:</span>
                                    &nbsp;We will lay out the details of different neural
                                    architecture and their applications in team recommendation,
                                    from autoencoder to graph neural networks
                                </li>
                                    <img src={require("../img/e2e.png")} alt="Top: Graph representation learning of skills." height="400"/>
                                    <p>Figure 2. Top: Graph representation learning of skills (<a target="_blank" href="https://doi.org/10.1145/3404835.3463105">Rad et al., 2021</a>). Bottom: End-to-End Graph neural team recommendation (<a target="_blank" href="">Ahmed et al., 2025</a>)</p>
                                <li className="justify-paragraph">
                                    <span className="fw-bold">Training Strategies:</span>
                                    &nbsp;In our tutorial, we will discuss the details of
                                    different negative sampling heuristics to draw virtually
                                    unsuccessful teams and streaming training strategy that put a
                                    chronological order on teams during training.
                                </li>
                                <img src={require("../img/flow-t.jpg")} alt="Streaming training strategy in neural-based team formation methods." height="350" />
                                <p>{" "}Figure 3. Streaming training strategy in neural-based team formation methods.</p>
                                <li className="justify-paragraph">
                                  <span className="fw-bold">
                                    Hands-On{" "}<a target="_blank" href="https://github.com/fani-lab/OpeNTF">OpeNTF</a>:
                                  </span>
                                    &nbsp;In our tutorial, we introduce publicly available
                                    libraries and tools for the task of team recommendation.
                                    Notably, we provide hands-on experience with{" "}
                                    <a target="_blank" href="https://github.com/fani-lab/OpeNTF">OpeNTF</a>
                                    , an open-source benchmark library for neural models.
                                </li>
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
            <section id="section-challeng"><span id="section-title-challeng" className="section-title">Challenges and New Perspectives (20 Minutes)</span>
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
                            <img src={require("../img/vivafemme.jpg")} alt="vivaFemme's loss regularization to mitigate gender bias in team recommendation." height="400" />
                            <p>Figure 4. vivaFemme's loss regularization to mitigate gender bias in team recommendation. [<a target="_blank" href="https://github.com/fani-lab/OpeNTF/tree/vivaFemme-bias24">code</a>][<a target="_blank" href="https://hosseinfani.github.io/res/papers/2024_BIAS_SIGIR_vivaFemme_Mitigating_Gender_Bias_in_Neural_Team_Recommendation_via_Female-Advocate_Loss_Regularization.pdf">paper</a>]</p>
                            <img src={require("../img/cl.png")} alt="Adaptive loss-based curricula to mitigate popularity bias." height="200" />
                            <p>Figure 5. Adaptive loss-based curricula to mitigate popularity bias.[<a target="_blank" href="https://github.com/fani-lab/OpeNTF/tree/cl-wsdm25">code</a>][<a target="_blank" href="https://hosseinfani.github.io/res/papers/2025_WSDM_Adaptive_Loss-based_Curricula_for_Neural_Team_Recommendation.pdf">paper</a>]</p>
                            <img src={require("../img/adila.png")} alt="Adila's reranking pipeline." height="300" />
                            <p>Figure 6. Adila's reranking pipeline. [<a target="_blank" href="https://github.com/fani-lab/Adila/tree/bias23">code</a>][<a target="_blank" href="https://hosseinfani.github.io/res/papers/2023_BIAS_ECIR_Bootless_Application_of_Greedy_Re-ranking_Algorithms_in_Fair_Neural_Team_Formation.pdf">paper</a>]</p>
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
            <section id="section-apps"><span id="section-title-apps" className="section-title">Applications (20 Minutes)</span>
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
                        <a target="_blank" href="https://www.linkedin.com/in/md-jamil-ahmed-545164247/" target="_blank">
                            <div className="presenter-img">
                                <img className="front" src={require("../img/jamil.jpg")} alt="Md Jamil Ahmed"/>
                                <img className="back" src={require("../img/cat_footballer.jpg")} alt="Cat Footballer"/>
                            </div>
                        </a>
                        <span className="ref-name fs-5"><a target="_blank" href="https://www.linkedin.com/in/md-jamil-ahmed-545164247/"
                                                           target="_blank">Md Jamil Ahmed</a></span>
                        <span className="text-muted fs-10">MSc Student<br/>School of Computer Science<br/>University of Windsor</span>

                    </div>
                    <div className="presenter">
                        <a target="_blank" href="https://www.linkedin.com/in/christine-wong-6828b0193/" target="_blank">
                            <div class="presenter-img">
                                <img class="front" src={require("../img/christine.jpg")} alt="Christine Wong"/>
                                <img class="back" src={require("../img/cardinal.jpg")} alt="Cardinal"/>
                            </div>
                        </a>
                        <span className="ref-name fs-5"><a target="_blank" href="https://www.linkedin.com/in/christine-wong-6828b0193/" target="_blank">Christine Wong</a></span>
                        <span className="text-muted fs-10">BSc Student<br/>School of Computer Science<br/>University of Windsor</span>
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
