import { useEffect } from "react";
import "./Home.css";

function Home() {
  let offsets = [];
  useEffect(() => {
    window.addEventListener("load", onWindowLoaded);
    window.addEventListener("scroll", onWindowScroll);
  }, []);

  const onWindowLoaded = () => {
    const ids = [
      "location",
      "abstract",
      "audience",
      "prereq",
      "outline",
      "presenters",
      "download",
    ];
    const topMargin =
      parseInt(getComputedStyle(window.document.body).fontSize) * 6;
    offsets = ids.map((id) => {
      const offset =
        document.getElementById("section-title-" + id)?.offsetTop -
        topMargin -
        15;
      return {
        id,
        offset,
        endOffset:
          offset + document.getElementById(`section-${id}`).offsetHeight - 20,
      };
    });
  };

  const scrollToItem = (e) => {
    const topMargin =
      parseInt(getComputedStyle(window.document.body).fontSize) * 6;
    const scroll = document.getElementById(`footnote-${e.target.id}`).offsetTop;
    window.scrollTo({
      top: scroll - topMargin,
    });
  };

  const activeMenuItem = (id) => {
    document.querySelector(".side-menu li.active")?.classList?.remove("active");
    document.getElementById(id).classList.add("active");
  };

  const activeSection = (id) => {
    document
      .querySelector(".home-content section.active")
      ?.classList?.remove("active");
    document.getElementById(`section-${id}`).classList.add("active");
  };

  const onWindowScroll = (e) => {
    e.preventDefault();
    const currentPosition = window.pageYOffset;
    for (let index = 0; index < offsets.length; index++) {
      const section = offsets[index];
      if (
        section.offset <= currentPosition &&
        section.endOffset > currentPosition
      ) {
        activeMenuItem(section.id);
        activeSection(section.id);
        break;
      }
    }
  };

  return (
    <div className="home-content">
      <section id="section-location" class="active">
        <span id="section-title-location" className="section-title">
          Time and Location
        </span>
        <div className="section-body" style={{ textAlign: "center" }}>
          <b>
            11:00am-12:30pm (CEST) <br /> Thursday, July 4, 2024, Room 12
          </b>{" "}
          <br />
          <b>
            <a href="https://meet.google.com/ikh-fnin-dns">
              https://meet.google.com/ikh-fnin-dns
            </a>
          </b>
        </div>
      </section>
      <section id="section-abstract">
        <span id="section-title-abstract" className="section-title">
          Abstract
        </span>
        <div className="section-body justify-paragraph">
          Collaborative team recommendation involves selecting users with
          certain skills to form a team who will, more likely than not,
          accomplish a complex task successfully. To automate the traditionally
          tedious and error-prone manual process of team formation, researchers
          from several scientific spheres have proposed methods to tackle the
          problem. In this tutorial, while providing a taxonomy of team
          recommendation works based on their algorithmic approaches to model
          skilled users in collaborative teams, we perform a comprehensive and
          hands-on study of the graph-based approaches that comprise the
          mainstream in this field, then cover the neural team recommenders as
          the cutting-edge class of approaches. Further, we provide unifying
          definitions, formulations, and evaluation schema. Last, we introduce
          details of training strategies, benchmarking datasets, and open-source
          tools, along with directions for future works.
        </div>
      </section>
      <section id="section-audience">
        <span id="section-title-audience" className="section-title">
          Target Audience
        </span>
        <div className="section-body justify-paragraph">
          Team recommendation problem falls under social information retrieval
          (Social IR) where we seek to find the right group of skillful users to
          solve the tasks at hand or only with the assistance of social
          resources. In this tutorial,
          <ul>
            <li className="justify-paragraph">
              we target beginner or intermediate researchers, industry
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
              different models and readily pick the most suitable one for their
              application to form collaborative teams of skilled users whose
              success is almost surely guaranteed.
            </li>
          </ul>
        </div>
      </section>
      <section id="section-prereq">
        <span id="section-title-prereq" className="section-title">
          Prerequisite Knowledge
        </span>
        <div className="section-body justify-paragraph">
          The target audience needs to be familiar with graph theory and machine
          learning. Where appropriate, the tutorial will not make any
          assumptions about the audience’s knowledge on more advanced
          techniques. As such, sufficient details will be provided as
          appropriate so that the content will be accessible and understandable
          to those with a fundamental understanding of such principles.
        </div>
      </section>
      <section id="section-outline">
        <span id="section-title-outline" className="section-title">
          Outline
        </span>
        <div className="section-body">
          <span className="d-block w-100 justify-paragraph">
            From Figure 1 (below), we begin to introduce intuitive definitions
            of a team and some representative, historical to modern and
            state-of-the-art methods for solving the team recommendation
            problem, motivating the importance of the problem, followed by a
            novel taxonomy of computational methods, as explained hereafter.
          </span>
          <img
            src={require("../img/taxonomy.jpg")}
            alt="Taxonomy of team recommendation methods."
            height="300"
          />
          <p> Figure 1. Taxonomy of team recommendation methods.</p>
          <div className="outline-topic">
            <span className="section-date">35 minutes</span>
            <span className="fw-bold text-uppercase h5">
              Search-based Heuristics
            </span>
            <span className="d-block w-100 justify-paragraph">
              This section provides an overview of the graph-based approaches in
              team formation methods. Operations Research-based methods,
              although conceiving the foremost computational models, overlooked
              the organizational and social ties among users and are hence
              excluded in our tutorial.
            </span>
            <div className="topic-item">
              <ul>
                <li className="justify-paragraph">
                  <span className="fw-bold">
                    Subgraph Optimization Objectives:
                  </span>
                  &nbsp;In our tutorial, we formalized more than 13 objectives
                  in a unified framework with integrated notations for better
                  readability and fostering conventions in this realm.
                </li>
                <li className="justify-paragraph">
                  <span className="fw-bold">
                    Subgraph Optimization Techniques:
                  </span>
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
                <li>
                  <a href="https://doi.org/10.1016/j.eswa.2021.114886">
                    A unified framework for effective team formation in social
                    networks
                  </a>{" "}
                  (Selvarajah et al., 2021)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/3459637.3481969">
                    RW-Team: Robust Team Formation using Random Walk
                  </a>{" "}
                  (Nemec et al., 2021)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/3465399">
                    A Comprehensive Review and a Taxonomy Proposal of Team
                    Formation Problems
                  </a>{" "}
                  (Juarez et al., 2021)
                </li>
                <li>
                  <a href="https://doi.org/10.1109/TKDE.2020.2985376">
                    Effective Keyword Search Over Weighted Graphs
                  </a>{" "}
                  (Kargar et al., 2020)
                </li>
                <li>
                  <a href="https://doi.org/10.1093/comjnl/bxw088">
                    Forming Grouped Teams with Efficient Collaboration in Social
                    Networks
                  </a>{" "}
                  (Huang et al., 2017)
                </li>
                <li>
                  <a href="https://doi.org/10.5441/002/edbt.2017.54">
                    Authority-based Team Discovery in Social Networks
                  </a>{" "}
                  (Zihayat et al., 2017)
                </li>
                <li>
                  <a href="https://doi.org/10.1109/WI-IAT.2014.112">
                    Two-Phase Pareto Set Discovery for Team Formation in Social
                    Networks
                  </a>{" "}
                  (Zihayat et al., 2014)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/2488388.2488482">
                    Towards realistic team formation in social networks based on
                    densest subgraphs
                  </a>{" "}
                  (Rangapuram et al., 2013)
                </li>
                <li>
                  <a href="https://doi.org/10.1137/1.9781611972825.15">
                    Multi-skill Collaborative Teams based on Densest Subgraphs
                  </a>{" "}
                  (Gajewar et al., 2012)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/2339530.2339690">
                    Capacitated team formation problem on social networks
                  </a>{" "}
                  (Datta et al., 2012)
                </li>
                <li>
                  <a href="https://doi.org/10.1109/ICDMW.2011.28">
                    An Effective Expert Team Formation in Social Networks Based
                    on Skill Grading
                  </a>{" "}
                  (Farhadi et al., 2011)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/2063576.2063718">
                    Discovering top-k teams of experts with/without a leader in
                    social networks
                  </a>{" "}
                  (Kargar et al., 2011)
                </li>
                <li>
                  <a href="https://doi.org/10.1109/SocialCom.2010.12">
                    Team Formation for Generalized Tasks in Expertise Social
                    Networks
                  </a>{" "}
                  (Li et al., 2010)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/1557019.1557074">
                    Finding a team of experts in social networks
                  </a>{" "}
                  (Lappas et al., 2009)
                </li>
              </ul>
            </div>
          </div>
          <div className="outline-topic">
            <span className="section-date">35 minutes</span>
            <span className="fw-bold text-uppercase h5">
              Learning-based Heuristics
            </span>
            <span className="d-block w-100 justify-paragraph">
              We will then explain the learning-based methods, which has been
              mostly based on neural models. Learning-based methods bring
              efficiency while enhancing efficacy due to the inherently
              iterative and online learning procedure, and can address the
              limitations of search-based solutions with respect to scalability,
              as well as dynamic expert networks (
              <a href="https://dl.acm.org/doi/10.1145/3589762">
                Rad, R. et al., 2023
              </a>
              ;{" "}
              <a href="https://dl.acm.org/doi/10.1145/3340531.3412140">
                Rad, R.H., et al., 2020
              </a>
              ).
            </span>
            <div className="topic-item">
              <ul>
                <li className="justify-paragraph">
                  <span className="fw-bold">Neural Architectures:</span>
                  &nbsp;We will lay out the details of different neural
                  architecture and their applications in team recommendation,
                  from autoencoder to graph neural networks.
                </li>
                <li className="justify-paragraph">
                  <span className="fw-bold">Training Strategies:</span>
                  &nbsp;In our tutorial, we will discuss the details of
                  different negative sampling heuristics to draw virtually
                  unsuccessful teams and streaming training strategy that put a
                  chronological order on teams during training.
                </li>
                <img
                  src={require("../img/flow-t.jpg")}
                  alt="Streaming training strategy in neural-based team formation methods."
                  height="300"
                />
                <p>
                  {" "}
                  Figure 2. Streaming training strategy in neural-based team
                  formation methods.
                </p>
                <li className="justify-paragraph">
                  <span className="fw-bold">
                    Hands-On{" "}
                    <a href="https://github.com/fani-lab/OpeNTF">
                      <i>OpeNTF</i>
                    </a>
                    :
                  </span>
                  &nbsp;In our tutorial, we introduce publicly available
                  libraries and tools for the task of team recommendation.
                  Notably, we provide hands-on experience with{" "}
                  <a href="https://github.com/fani-lab/OpeNTF">
                    <i>OpeNTF</i>
                  </a>
                  , an open-source benchmark library for neural models.​
                </li>
                <img
                  src={require("../img/bnn.jpg")}
                  alt="Bayesian neural network (bnn), one of OpeNTF's supported neural models."
                  height="300"
                />
                <p>
                  {" "}
                  Figure 3. Bayesian neural network (bnn), one of{" "}
                  <a href="https://github.com/fani-lab/OpeNTF">
                    <i>OpeNTF</i>
                  </a>
                  's supported neural models.
                </p>
              </ul>
            </div>
            <div className="topic-item">
              <span className="fw-bold text-uppercase h6">Reading List</span>
              <ul>
                <li>
                  <a href="https://link.springer.com/chapter/10.1007/978-3-031-56027-9_20">
                    A Streaming Approach to Neural Team Formation Training
                  </a>{" "}
                  (Fani et al., 2024)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/3589762">
                    A Variational Neural Architecture for Skill-based Team
                    Formation
                  </a>{" "}
                  (Rad et al., 2023)
                </li>
                <li>
                  <a href="https://doi.org/10.1109/IJCNN54540.2023.10191717">
                    Transfer Learning with Graph Attention Networks for Team
                    Recommendation
                  </a>{" "}
                  (Kaw et al., 2023)
                </li>
                <li>
                  <a href="https://doi.org/10.1007/s10791-023-09421-6">
                    Learning heterogeneous subgraph representations for team
                    discovery
                  </a>{" "}
                  (Nguyen et al., 2023)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/3511808.3557526">
                    OpeNTF: A Benchmark Library for Neural Team Formation
                  </a>{" "}
                  (Dashti et al., 2022)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/3511808.3557590">
                    Effective Neural Team Formation via Negative Samples
                  </a>{" "}
                  (Dashti et al., 2022)
                </li>
                <li>
                  <a href="https://doi.org/10.48786/edbt.2022.37">
                    A Neural Approach to Forming Coherent Teams in Collaboration
                    Networks
                  </a>{" "}
                  (Rad et al., 2022)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/3501247.3531578">
                    Subgraph Representation Learning for Team Mining
                  </a>{" "}
                  (Rad et al., 2022)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/3404835.3463105">
                    Retrieving Skill-Based Teams from Collaboration Networks
                  </a>{" "}
                  (Rad et al., 2021)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/3459637.3481992">
                    PyTFL: A Python-based Neural Team Formation Toolkit
                  </a>{" "}
                  (Rad et al., 2021)
                </li>
                <li>
                  <a href="https://doi.org/10.3389%2Ffdata.2019.00014">
                    Deep Neural Networks for Optimal Team Composition
                  </a>{" "}
                  (Sapienza et al., 2019)
                </li>
                <li>
                  <a href="https://dl.acm.org/doi/10.1145/3340531.3412140">
                    Learning to Form Skill-based Teams of Experts
                  </a>{" "}
                  (Rad et al., 2019)
                </li>
                <li>
                  <a href="https://dl.acm.org/doi/10.1145/3097983.3098036">
                    metapath2vec: Scalable Representation Learning for
                    Heterogeneous Networks
                  </a>{" "}
                  (Dong et al., 2017)
                </li>
                <li>
                  <a href="https://www.semanticscholar.org/paper/Representation-Learning-on-Graphs%3A-Methods-and-Hamilton-Ying/ecf6c42d84351f34e1625a6a2e4cc6526da45c74">
                    Representation Learning on Graphs: Methods and Applications
                  </a>{" "}
                  (Hamilton et al., 2017)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/1935826.1935914">
                    Supervised Random Walks: Predicting and Recommending Links
                    in Social Networks
                  </a>{" "}
                  (Backstrom et al., 2011)
                </li>
              </ul>
            </div>
            <div className="topic-item">
              <span className="fw-bold expand-button"></span>
              <span className="d-block w-100 justify-paragraph"></span>
            </div>
          </div>
          <div className="outline-topic">
            <span className="section-date">20 minutes</span>
            <span className="fw-bold text-uppercase h5">
              Challenges and New Perspectives
            </span>
            <div className="topic-item">
              <span className="fw-bold">
                <a href="https://github.com/fani-lab/Adila">
                  <i>Adila</i>
                </a>
                : Fair and Diverse Team Recommendation
              </span>
              <span className="d-block w-100 justify-paragraph">
                The primary focus of existing team recommendation methods is the
                maximization of the success rate for the recommended teams,
                largely ignoring diversity in the recommended users. In our
                tutorial, we introduce notions of fairness and{" "}
                <a href="https://github.com/fani-lab/Adila">
                  <i>Adila</i>
                </a>
                , that enables further post-processing reranking refinements to
                reassure the desired fair outcome and explore the synergistic
                trade-offs between notions of fairness and success rate for the
                proposed solutions.
              </span>
              <img
                src={require("../img/flow.jpg")}
                alt="Adila's pipeline architecture."
                height="300"
              />
              <p>
                {" "}
                Figure 4.{" "}
                <a href="https://github.com/fani-lab/Adila">
                  <i>Adila</i>
                </a>
                's pipeline architecture.
              </p>
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
                <li>
                  <a href="https://doi.org/10.1007/978-3-031-37249-0_9">
                    Bootless Application of Greedy Re-ranking Algorithms in Fair
                    Neural Team Formation
                  </a>{" "}
                  (Loghmani et al., 2023)
                </li>
                <li>
                  <a href="https://doi.org/10.1016/j.ipm.2021.102707">
                    Fair Top-k Ranking with Multiple Protected Groups
                  </a>{" "}
                  (Zehlike et al., 2022)
                </li>
                <li>
                  <a href="https://doi.org/10.1609/aaai.v36i11.21445">
                    Has ceo gender bias really been fixed? adversarial attacking
                    and improving gender fairness in image search
                  </a>{" "}
                  (Feng et al., 2022)
                </li>
                <li>
                  <a href="http://papers.nips.cc/paper_files/paper/2022/hash/cdd0640218a27e9e2c0e52e324e25db0-Abstract-Conference.html">
                    Fair Ranking with Noisy Protected Attributes
                  </a>{" "}
                  (Mehrotra et al., 2022)
                </li>
                <li>
                  <a href="https://link.springer.com/article/10.1007/s42452-020-2801-5">
                    Challenges and barriers in virtual teams: a literature
                    review
                  </a>{" "}
                  (Morrison et al., 2020)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/3292500.3330691">
                    Fairness-Aware Ranking in Search & Recommendation Systems
                    with Application to LinkedIn Talent Search
                  </a>{" "}
                  (Geyik et al., 2019)
                </li>
                <li>
                  <a href="https://doi.org/10.1145/1460563.1460633">
                    Who collaborates successfully? prior experience reduces
                    collaboration barriers in distributed interdisciplinary
                    research
                  </a>{" "}
                  (Cummings et al., 2008)
                </li>
              </ul>
            </div>
          </div>
        </div>
      </section>
      <section id="section-presenters">
        <span id="section-title-presenters" className="section-title">
          Presenters
        </span>
        <div className="section-body d-flex p-3 direction-row justify-content-between">
          <div className="presenter">
            <a
              href="https://www.linkedin.com/in/mahdis-saeedi-b80b8321a/"
              target="_blank"
            >
              <div class="presenter-img">
                <img
                  class="front"
                  src={require("../img/mahdis.jpg")}
                  alt="Mahdis Saeedi"
                />
                <img
                  class="back"
                  src={require("../img/flamingo.jpg")}
                  alt="Flamingo"
                />
              </div>
            </a>
            <span className="ref-name fs-5">Mahdis Saeedi</span>
            <span className="text-muted fs-6 fst-italic">
              University of Windsor
            </span>
          </div>
          <div className="presenter">
            <a
              href="https://www.linkedin.com/in/christine-wong-6828b0193/"
              target="_blank"
            >
              <div class="presenter-img">
                <img
                  class="front"
                  src={require("../img/christine.jpg")}
                  alt="Christine Wong"
                />
                <img
                  class="back"
                  src={require("../img/cardinal.jpg")}
                  alt="Cardinal"
                />
              </div>
            </a>
            <span className="ref-name fs-5">Christine Wong</span>
            <span className="text-muted fs-6 fst-italic">
              University of Windsor
            </span>
          </div>
          <div className="presenter">
            <a href="http://hosseinfani.github.io/" target="_blank">
              <div class="presenter-img">
                <img
                  class="front"
                  src={require("../img/hossein.jpg")}
                  alt="Hossein Fani"
                />
                <img
                  class="back"
                  src={require("../img/sloth.jpg")}
                  alt="Sloth"
                />
              </div>
            </a>
            <span className="ref-name fs-5">Hossein Fani</span>
            <span className="text-muted fs-6 fst-italic">
              University of Windsor
            </span>
          </div>
        </div>
      </section>
      <section id="section-download">
        <span id="section-title-download" className="section-title">
          Materials
        </span>
        <div className="section-body p-2">
          <div class="p-2">
            <a
              target="_blank"
              href="https://hosseinfani.github.io/res/papers/2024_UMAP_Collaborative_Team_Recommendation_for_Skilled_Users_Objectives_Techniques_and_New_Perspectives.pdf"
            >
              Full Outline
            </a>
          </div>
          <div class="p-2">
            <a href="javascript:void(0)" style={{ cursor: "default" }}>
              Presentation Slides (Upcoming)
            </a>
          </div>
          <div class="p-2">
            <a href="javascript:void(0)" style={{ cursor: "default" }}>
              Presentation Video (Upcoming)
            </a>
          </div>
        </div>
      </section>
    </div>
  );
}

export default Home;
