#ifndef bitUbtye_HPP_INCLUDED
#define bitUbtye_HPP_INCLUDED

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
    memset(bits, 0, sizeof(bool) * 8);
    for (auto i : boolarr){
        bits[idxu]=static_cast<bool>(i);
        idxu++;
        if(idxu==8){
            bytearr.push_back(ToByte(bits));
            idxu=0;
            memset(bits, 0, sizeof(bool) * 8);
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

inline 
bool getbits(std::vector<unsigned char>& bytearr,std::size_t idx,std::size_t idy){
    int id_byte0=int(idx/8.0);
    int id_byte01=idx%8;
    int id_byte1=int(idy/8.0);
    int id_byte11=idy%8;    
}


#endif