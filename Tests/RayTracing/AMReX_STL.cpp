#include <AMReX_STL.H>

using namespace amrex;
using namespace std;

vector<string> STL::detail::split (string s, string delimiter) noexcept
{
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    string token;
    vector<string> res;

    while ((pos_end = s.find (delimiter, pos_start)) != string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}

vector<Triangle> STL::read_ascii_stl (const string& filename) noexcept {
    string line;
    ifstream myfile(filename);
    vector<Triangle> triangles;
    if (myfile.is_open()) {
        while ( getline (myfile, line) ) {
            if (line.find("facet normal") != string::npos) { // each instance is 1 triangle
                XDim3 coords[3];
                getline (myfile, line);  // skip "outer loop"
                for (int i = 0; i < 3; ++i) {  // always 3 vertices
                    getline (myfile, line);
                    auto vals = detail::split(line, " ");
                    size_t end_pos = vals.size()-1;
                    AMREX_ASSERT(end_pos > 2);  // last three entries are coordinates
                    coords[i].x = Real(stod(vals[end_pos-2]));
                    coords[i].y = Real(stod(vals[end_pos-1]));
                    coords[i].z = Real(stod(vals[end_pos  ]));
                }
                triangles.emplace_back(Triangle{coords[0], coords[1], coords[2]});
            }
        }
        myfile.close();
    }
    else {
        amrex::Abort("Unable to open mesh file");
    }

    return triangles;
}
