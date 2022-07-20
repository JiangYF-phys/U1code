#ifndef repmap_hpp
#define repmap_hpp

#include <stdio.h>
#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <assert.h>

using namespace std;

// rep map

struct repmap {
public:
    int j1, j2, j, n1, n2, n;
    int bgn, len, end;
    
    repmap(int j1=0, int j2=0, int j=0, int n1=0, int n2=0, int n=0, int bgn=0, int len=0, int end=0);
	~repmap();

    bool operator ==(const repmap& map1) const;
    bool operator <(const repmap& map1) const;
    void todisk(ofstream& out) const;
    void fromdisk(ifstream& in);
};

int searchmap(const vector<repmap> &map, const int &j1, const int &j2, const int &j, const int &n1, const int &n2, const int &n);

void maptodisk(const vector<repmap> &map, const string filename);
void mapfromdisk(vector<repmap> &map, const string filename);

ostream& operator <<(ostream& out, const repmap& map);

#endif /* repmap_hpp */
