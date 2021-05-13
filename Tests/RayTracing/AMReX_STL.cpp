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

vector<Triangle> STL::read_stl (const string& filename) noexcept {
    string line;
    ifstream myfile(filename);
    vector<Triangle> triangles;
    if (myfile.is_open()) {
        getline(myfile, line);
            if (line.find("solid ") == 0) {
                triangles = read_ascii_stl(filename);
            }
            else {
                triangles = read_binary_stl(filename);
            }
    }
    else {
        amrex::Abort("Unable to open mesh file");
    }

    return triangles;
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

vector<Triangle> STL::read_binary_stl (const string& filename) noexcept {
    string line;
    ifstream myfile(filename, std::ios::in | std::ios::binary);
    vector<Triangle> triangles;
    if (myfile.is_open()) {
        char header[80];
        myfile.read(&header[0], 80*sizeof(char));

        std::uint32_t num_triangles;
        myfile.read((char*)&num_triangles, sizeof(std::uint32_t));

        STL::BinaryTriangle tri;
        for (std::size_t i = 0; i < num_triangles; ++i)
        {
            constexpr std::size_t tri_bytes = 50;
            myfile.read((char*)&tri, tri_bytes);
            Triangle new_tri;

            new_tri.p0.x = tri.p0[0];
            new_tri.p0.y = tri.p0[1];
            new_tri.p0.z = tri.p0[2];

            new_tri.p1.x = tri.p1[0];
            new_tri.p1.y = tri.p1[1];
            new_tri.p1.z = tri.p1[2];

            new_tri.p2.x = tri.p2[0];
            new_tri.p2.y = tri.p2[1];
            new_tri.p2.z = tri.p2[2];

            triangles.push_back(new_tri);
        }
        myfile.close();
    }
    else {
        amrex::Abort("Unable to open mesh file");
    }

    return triangles;
}
