#include "repmap.hpp"

repmap::repmap(int j1, int j2, int j, int n1, int n2, int n, int bgn, int len, int end) {
    this->j = j; this->n = n;
    this->j1 = j1 ; this->j2  = j2;
    this->n1 = n1 ; this->n2  = n2;
    this->bgn= bgn; this->len = len; this->end = end;
}

repmap::~repmap() {
}

bool repmap::operator ==(const repmap& map1) const {
    if (this->j==map1.j && this->n==map1.n) {
        if (this->j1==map1.j1 && this->j2==map1.j2 && this->n1==map1.n1 && this->n2==map1.n2) {
            return true;
        }
    }
	return false;
}

bool repmap::operator< (const repmap& m2) const {
    if (j<m2.j) {
        return true;
    } else if (j==m2.j) {
        if (n<m2.n) {
            return true;
        } else if (n==m2.n) {
            if (j1<m2.j1) {
                return true;
            } else if (j1==m2.j1) {
                if (j2<m2.j2) {
                    return true;
                } else if (j2==m2.j2) {
                    return n1<m2.n1; // n2=n-n1
                }
            }
        }
    }
    return false;
}

int searchmap(const vector<repmap> &map, const int &j1, const int &j2, const int &j, const int &n1, const int &n2, const int &n) {
    assert( j1+j2==j && n1+n2==n );
    repmap map1(j1,j2,j,n1,n2,n,0,0,0);
    
    std::pair< vector<repmap>::const_iterator, vector<repmap>::const_iterator > range=equal_range(map.begin(), map.end(), map1);
    if (range.first!=map.end()) {
        return range.first-map.begin();
    }
    return -1;
}

ostream& operator<<(ostream& out, const repmap& map) {
    out << endl;
    cout << "{" << map.j << "," << map.j1 << "," << map.j2 << "}" << endl;
    cout << "{" << map.n << "," << map.n1 << "," << map.n2 << "}" << endl;
    cout << "{" << map.bgn << "," << map.len << "," << map.end << "}" << endl;
    return out;
}

void repmap::todisk(ofstream& out) const {
    out.write((char *) (&j), sizeof(j));
    out.write((char *) (&j1), sizeof(j1));
    out.write((char *) (&j2), sizeof(j2));
    out.write((char *) (&n), sizeof(n));
    out.write((char *) (&n1), sizeof(n1));
    out.write((char *) (&n2), sizeof(n2));
    out.write((char *) (&bgn), sizeof(bgn));
    out.write((char *) (&len), sizeof(len));
    out.write((char *) (&end), sizeof(end));
}

void repmap::fromdisk(ifstream& in) {
    in.read((char *) (&j), sizeof(j));
    in.read((char *) (&j1), sizeof(j1));
    in.read((char *) (&j2), sizeof(j2));
    in.read((char *) (&n), sizeof(n));
    in.read((char *) (&n1), sizeof(n1));
    in.read((char *) (&n2), sizeof(n2));
    in.read((char *) (&bgn), sizeof(bgn));
    in.read((char *) (&len), sizeof(len));
    in.read((char *) (&end), sizeof(end));
}

void maptodisk(const vector<repmap> &map, const string filename) {
    ofstream out(filename.c_str(), ios::out | ios::binary | ios::trunc);
    long size;
    size = map.size();
    out.write((char*) (&size), sizeof(size));
    for (size_t i=0; i<map.size(); ++i) {
        map[i].todisk(out);
    }
    out.close();
}

void mapfromdisk(vector<repmap> &map, const string filename) {
    ifstream in(filename.c_str(), ios::out | ios::binary);
    long size;
    in.read((char*) (&size), sizeof(size));
    map.resize(size);
    for (int i=0; i<size; ++i) {
        map[i].fromdisk(in);
    }
    in.close();
}
