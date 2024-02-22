import { Outlet } from "react-router-dom";

import Menu from './components/Menu';
import Sidebar from './components/Sidebar';

function App() {
  return (
    <div className='container-fluid'>
      <div className='row'>
        <Menu />
      </div>
      <div className='row main-content'>
        <div className='col-4 pr-5 pr-md-0 offset-md-1 col-md-3 position-fixed'>
          <Sidebar />
        </div>
        <div className='col-8 offset-4 col-md-7'>
          <Outlet />
        </div>
      </div>
    </div>
  );
}

export default App;
