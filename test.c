#include <stdio.h>
#include "lua/lua.h"
#include "lua/lualib.h"
#include "lua/lauxlib.h"
static int add(lua_State *L)
{
    int a,b,c;
    a = lua_tonumber(L,1);
    b = lua_tonumber(L,2);
    c = a+b;
    lua_pushnumber(L,c);
    printf("test hello!!!\r\n");
    return 1;
}
static const struct luaL_Reg lib[] =
{
    {"testadd",add},
    {NULL,NULL}
};

int luaopen_testlib(lua_State *L)
{
    luaL_register(L,"testlib",lib);
    return 1;
}