function Menu() {
  return (
    <div className="d-flex flex-column justify-content-center top-menu">
      <div className="row">
        <div className="center-block title-box">
          {/*<div className="text-left" style={*/}
          {/*  {position: "absolute",*/}
          {/*    top: 0,*/}
          {/*    left: "50%",*/}
          {/*    transform: "translateX(-50%)",*/}
          {/*    textAlign: "left",*/}
          {/*    whiteSpace: "nowrap"}}>*/}
          {/*  A Tutorial on*/}
          {/*</div>*/}
          <div className="text-center">
            <span className="fs-2 title-text">Neural Shifts in Collaborative Team Recommendation
            </span>
          </div>
          <div className="text-center" >
            <span className="title-text">
              <a className="plain-link" href="https://fani-lab.github.io/"><img className="p-0" src={require("../img/logo.jpg")} alt="fani's lab logo" height={"30"} /> fani-lab.github.io</a>&nbsp;&nbsp;
              <a className="plain-link" href="https://cikm2025.org/"><img className="p-0"  src={require("../img/cikm-logo.png")} alt="cikm conference logo" height={"50"} />cikm2025.org</a>
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Menu;
