import './Sidebar.css'

function Sidebar() {
    const topMargin = parseInt(getComputedStyle(window.document.body).fontSize) * 6;

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
            <li onClick={menuClicked} id="prereq">Prerequisite Knowledge</li>
            <li onClick={menuClicked} id="outline">Outline</li>
            <li onClick={menuClicked} id="presenters">Presenters</li>
            <li onClick={menuClicked} id="download">Materials</li>
        </ul>
    </ >;
}

export default Sidebar;