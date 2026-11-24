#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
int main(int c,char**v){uint64_t hh=1469598103934665603ULL;for(int k=1;k<c;k++){FILE*f=fopen(v[k],"rb");if(!f)return 1;uint64_t h=1469598103934665603ULL;int ch;while((ch=getc_unlocked(f))!=EOF){h^=(uint8_t)ch;h*=1099511628211ULL;}fclose(f);hh^=h+(uint64_t)k;}printf("%016llx
",(unsigned long long)hh);return 0;}
