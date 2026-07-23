import Layout from "./Layout.jsx";

import Home from "./Home";
import Library from "./Library";
import ProjectDetail from "./ProjectDetail";
import PaperDetail from "./PaperDetail";
import Styleguide from "./dev/Styleguide";
import PaperDetailMock from "./dev/PaperDetailMock";

import { BrowserRouter as Router, Route, Routes, useLocation } from 'react-router-dom';

const PAGES = {

    Home: Home,

    Library: Library,

}

function _getCurrentPage(url) {
    if (url.endsWith('/')) {
        url = url.slice(0, -1);
    }
    let urlLastPart = url.split('/').pop();
    if (urlLastPart.includes('?')) {
        urlLastPart = urlLastPart.split('?')[0];
    }

    const pageName = Object.keys(PAGES).find(page => page.toLowerCase() === urlLastPart.toLowerCase());
    return pageName || Object.keys(PAGES)[0];
}

// Create a wrapper component that uses useLocation inside the Router context
function PagesContent() {
    const location = useLocation();
    const currentPage = _getCurrentPage(location.pathname);

    // A0 dev routes render standalone: the old Layout shell is what A1
    // replaces, and wrapping new-token screens in it would muddy review.
    if (location.pathname.startsWith('/styleguide')) {
        return (
            <Routes>
                <Route path="/styleguide" element={<Styleguide />} />
                <Route path="/styleguide/paper" element={<PaperDetailMock />} />
            </Routes>
        );
    }

    return (
        <Layout currentPageName={currentPage}>
            <Routes>

                    <Route path="/" element={<Home />} />


                <Route path="/Home" element={<Home />} />

                <Route path="/library" element={<Library />} />

                <Route path="/projects/:id" element={<ProjectDetail />} />

                <Route path="/papers/:id" element={<PaperDetail />} />

            </Routes>
        </Layout>
    );
}

export default function Pages() {
    return (
        <Router>
            <PagesContent />
        </Router>
    );
}
