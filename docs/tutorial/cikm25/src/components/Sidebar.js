import './Sidebar.css'

function Sidebar() {
    const topMargin = 10 //parseInt(getComputedStyle(window.document.body).fontSize) * 10;

    const menuClicked = (e) => {
        // document.getElementById(`section-${e.target.id}`).scrollIntoView()
        const scroll = document.getElementById(`section-title-${e.target.id}`).offsetTop;
        window.scrollTo({
            top: scroll - topMargin
        })
    }
    return <>
        <ul className="list-unstyled text-end side-menu">
            <li onClick={menuClicked} id="location" className='active'>Time and Location</li>
            <li onClick={menuClicked} id="abstract">Abstract</li>
            <li onClick={menuClicked} id="audience">Target Audience</li>
            <li onClick={menuClicked} id="outline">Outline</li>
            <li onClick={menuClicked} id="searchbased">Pioneering Techniques</li>
            <li onClick={menuClicked} id="learnbased"> Neural Team Recommendation</li>
            <li onClick={menuClicked} id="refinement">Team Refinement</li>
            <li onClick={menuClicked} id="challeng">Challenges and New Perspectives</li>
            <li onClick={menuClicked} id="apps">Applications</li>  <li onClick={menuClicked} id="handson">Hands-on OpeNTF</li>
            <li onClick={menuClicked} id="presenters">Presenters</li>
        </ul>
    </ >;
}

export default Sidebar;