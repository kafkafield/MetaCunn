#include <unordered_map>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

#include <luaT.h>
#include <lua.hpp> 

using namespace std;

lua_State* L;

struct inputSize {
	int bs; int ni; int no; int kw; int kh; int iw; int ih; int dw; int dh;
};

struct outputSize {
	int outMod;
	int gradInputMod;
	int gradParaMod;
	void print()
	{
		std::cout << outMod << " " << gradInputMod << " " << gradParaMod << std::endl;
	}
};

struct hash_func {
	size_t operator()(const inputSize &a) const  
    {  
         return (a.bs * 2333 + a.ni * 233 + a.no * 23 + a.kw * 23333 + a.kh * 233333 - a.iw * 2333 - a.ih * 233 - a.dw * 32 - a.dh);
    } 
};

struct cmp_func {
	bool operator()(const inputSize &a, const inputSize &b) const  
    {  
         if(a.bs == b.bs && a.ni == b.ni && a.no == b.no && a.kw == b.kw && a.kh == b.kh && a.iw == b.iw && a.ih == b.ih && a.dw == b.dw && a.dh == b.dh)
         	return true;
         else
         	return false;
    }  
};

extern "C" {
	static int loadmap_c(lua_State* L);
	static int findmax_c(lua_State* L);
	int luaopen_metahard(lua_State* L);
}

outputSize loadmap(int bs, int ni, int no, int kw, int kh, int iw, int ih, int dw, int dh)
{
	unordered_map<inputSize, outputSize, hash_func, cmp_func> loadFile;
	ifstream file("data");
	if(! file.is_open())
	{
		cout << "Error opening file. " << endl;
		exit(1);
	}
	inputSize tempi;
	outputSize tempo;
	string line;
	while(getline(file, line))
	{
		istringstream iss(line);
		iss >> tempi.bs;
		iss >> tempi.ni;
		iss >> tempi.no;
		iss >> tempi.kw;
		iss >> tempi.kh;
		iss >> tempi.iw;
		iss >> tempi.ih;
		iss >> tempi.dw;
		iss >> tempi.dh;
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


static int loadmap_c(lua_State* L)
{
	int bs = lua_tonumber(L, 1);
	int ni = lua_tonumber(L, 2);
	int no = lua_tonumber(L, 3);
	int kw = lua_tonumber(L, 4);
	int kh = lua_tonumber(L, 5);
	int iw = lua_tonumber(L, 6);
	int ih = lua_tonumber(L, 7);
	int dw = lua_tonumber(L, 8);
	int dh = lua_tonumber(L, 9);
	outputSize targeto = loadmap(bs, ni, no, kw, kh, iw, ih, dw, dh);
	lua_pushnumber(L, targeto.outMod);
	lua_pushnumber(L, targeto.gradInputMod);
	lua_pushnumber(L, targeto.gradParaMod);
	return 3;
}

int findmax()
{
	int max = 0;
	int temp;
	string line;
	ifstream file("heihei");
	if(! file.is_open())
	{
		cout << "Error opening file. " << endl;
		exit(1);
	}
	getline(file, line);
	while(getline(file, line))
	{
		istringstream iss(line);
		iss >> temp;
		if(temp > max)
			max = temp;
	}
	return max;
}

static int findmax_c(lua_State* L)
{
	luapushnumber(L, findmax());
	return 1;
}

const luaL_Reg functions[] = {
  {"loadmap", loadmap_c},
  {"getMaxMemory", findmax_c},
  {NULL,  NULL},
};

int luaopen_metahard(lua_State *L)
{
    luaL_register(L,"metahard",functions);
    lua_pop(L, 1);
    return 1;
}
/*
int main() // for test
{
	/* initialize Lua *
	L = lua_open();

	/* load Lua base libraries *
	luaL_openlibs(L);

	/* register our function *
	lua_register(L, "loadmap", average);

	/* run the script *
	luaL_dofile(L, "loadmaptest.lua");

	/* cleanup Lua *
	lua_close(L);

	/* pause *
	printf( "Press enter to exit..." );
	getchar();
	return 0;
}
*/