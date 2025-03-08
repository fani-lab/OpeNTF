function Menu() {
  return (
    <div className="d-flex flex-column justify-content-center top-menu">
      <div className="row">
        <div className="center-block title-box">
          <div className="text-center">
            <span className="fs-2 title-text">Bridging Historical Subgraph Optimization and Modern Graph Neural Network in Team Recommendation
            </span>
          </div>
          <div className="text-center" >
            <span className="title-text">
              <a className="plain-link" href="https://fani-lab.github.io/"><img className="p-0" src={require("../img/logo.jpg")} alt="fani's lab logo" height={"30"} /> fani-lab.github.io</a>&nbsp;&nbsp;
              <a className="plain-link" href="https://www.wsdm-conference.org/2025/"><img className="p-0"  src={require("../img/wsdm-logo-green.png")} alt="wsdm conference logo" height={"30"} />www.wsdm-conference.org/2025/</a>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Menu;
