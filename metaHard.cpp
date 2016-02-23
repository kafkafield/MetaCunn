#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

using namespace std;

int main() // for test
{
	loadmap(128,3,96,11,11,128,128,1,1).print();
	loadmap(128,64,128,9,9,64,64,1,1,1).print();
	loadmap(128,128,128,9,9,32,32,1,1).print();
	loadmap(128,128,128,7,7,16,16,1,1).print();
	loadmap(128,384,384,3,3,13,13,1,1).print();
	return 0;
}

struct inputSize {
	int bs; int ni; int no; int kw; int kh; int iw; int ih; int dw; int dh;
};

struct outputSize {
	int outMod;
	int gradInputMod;
	int gradParaMod;
	void print()
	{
		cout << outMod << " " << gradInputMod << " " << gradParaMod << endl;
	}
};

struct hash {
	size_t operator()(const inputSize &a) const  
    {  
         return (a.bs * 2333 + a.ni * 233 + a.no * 23 + a.kw * 23333 + a.kh * 233333 - a.iw * 2333 - a.ih * 233 - a.dw * 32 - a.dh);
    } 
};

struct cmp {
	bool operator()(const inputSize &a, const inputSize &b) const  
    {  
         if(a.bs == b.bs && a.ni == b.ni && a.no == b.no && a.kw == b.kw && a.kh == b.kh && a.iw == b.iw && a.ih == b.ih && a.dw == b.dw && a.dh == b.dh)
         	return true;
         else
         	return false;
    }  
};

outputSize loadmap(int bs, int ni, int no, int kw, int kh, int iw, int ih, int dw, int dh)
{
	unordered_map<inputSize, outputSize, hash, cmp> loadFile;
	ifstream file("data");
	if(! file.is_opren())
	{
		cout << "Error opening file. " << endl;
		return;
	}
	inputSize tempi;
	outputSize tempo;
	while(! file.eof())
	{
		string line;
		file.getline(line, 100);
		istringstream iss(line);
		iss >> tempi.bs;
		iss >> tempi.ni;
		iss >> tempi.no;
		iss >> tempi.kw;
		iss >> tempi.kh;
		iss >> tempi.iw;
		iss >> tempi.ih;
		iss >> tempi.dw;
		iss >> tempi.hd;
		iss >> tempo.outMod;
		iss >> tempo.gradInputMod;
		iss >> tempo.gradParaMod;
		loadFile[tempi] = tempo; 
	}
	inputSize targeti;
	outputSize targeto;
	targeti.bs = bs;
	targeti.ni = ni;
	targeti.no = no;
	targeti.kw = kw;
	targeti.kh = kh;
	targeti.iw = iw;
	targeti.ih = ih;
	targeti.dw = dw;
	targeti.dh = dh;
	auto got = loadFile.find(targeti);
	if(got == loadFile.end())
	{
		targeto.outMod = targeto.gradInputMod = targeto.gradParaMod = 0;
	}
	else
	{
		targeto = got->second;
	}
	file.close();
	return targeto;
}