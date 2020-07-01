#pragma once

#include <vector>

struct Bits
{
    unsigned b0:1, b1:1, b2:1, b3:1, b4:1, b5:1, b6:1, b7:1;
};
union CBits
{
    Bits bits;
    unsigned char byte;
};

inline unsigned char ToByte(bool b[8])
{
    unsigned char c = 0;
    for (int i=0; i < 8; ++i)
        if (b[i])
            c |= 1 << i;
    return c;
}

inline void FromByte(unsigned char c, bool b[8])
{
    for (int i=0; i < 8; ++i)
        b[i] = (c & (1<<i)) != 0;
}

template <typename T>
inline 
void fillbits(std::vector<unsigned char>& bytearr,std::vector<T>& boolarr){
    int idxu=0;
    std::size_t filled=0;
    bool bits[8];
    std::fill(bits, bits+8,0);
    for (auto i : boolarr){
        bits[idxu]=static_cast<bool>(i);
        idxu++;
        if(idxu==8){
            bytearr.push_back(ToByte(bits));
            idxu=0;
            std::fill(bits, bits+8,0);
            filled+=8;
        }
    }
    if (filled< boolarr.size())
        bytearr.push_back(ToByte(bits));
}

inline 
bool getbit(bool& r, std::vector<unsigned char>& bytearr,std::size_t id){
    int id_byte0=int(id/8.0);
    if (id_byte0>=bytearr.size())
        return false;
    int id_byte1=id%8;
    bool b[8];
    FromByte(bytearr[id_byte0], b);
    r=b[id_byte1];
    return true;
}

// typename T 表示输出的bool数列boolarr类型 可以是bool/int/char等
// 将8bits pack的bytearr输出成unpack的bool数列，取值区间[idx,idy]
// 成功返回1，如果idy>bytearr.size截断并返回-1，返回0则错误结果不可靠
template <typename T>
inline 
int getbits(std::vector<T>& boolarr,std::vector<unsigned char>& bytearr,std::size_t idx,std::size_t idy){
    int id_byte0=int(idx/8.0);
    int r=1;
    int id_byte01=idx%8;
    if (idy>=bytearr.size()*8){
        idy=bytearr.size()*8-1;
        r=-1;
    }
    int id_byte1=int(idy/8.0);
    int id_byte11=idy%8;    
    if (id_byte0==id_byte1){
        if(id_byte11>=id_byte01){
        //get bits[id_byte01,id_byte11]
            bool b[8];
            FromByte(bytearr[id_byte0],b);
            for (int i=id_byte01;i<=id_byte11;i++){
                boolarr.push_back(static_cast<T>(b[i]));
            }
            return r;
        }
        else{
            // std::cout<<"kao"<<std::endl;
            return 0;
        }
    }
    else if (id_byte1>id_byte0){
        //get bits[(id_byte0,[id_byte01,8])]
        bool b[8];
        FromByte(bytearr[id_byte0],b);
        for (int i=id_byte01;i<8;i++){
            boolarr.push_back(static_cast<T>(b[i]));
        }
        if(id_byte1>id_byte0+1){
            //get bits[id_byte0+1,id_byte1-1]
            for(int j=id_byte0+1;j<id_byte1;j++){
                bool b[8];
                FromByte(bytearr[j],b);
                for (int i=0;i<8;i++){
                    boolarr.push_back(static_cast<T>(b[i]));
                }
            }
        }
        //get bits[(id_byte1,[0,id_byte11])]
        FromByte(bytearr[id_byte1],b);
        for (int i=0;i<=id_byte11;i++){
            boolarr.push_back(static_cast<T>(b[i]));
        }        
        return r;
    }
    else{
        // std::cout<<std::endl<<id_byte0<<" kao2 "<<id_byte1<<std::endl;
        // std::cout<<idx<<" kao2 "<<idy<<std::endl;
        return 0;
    }        
}