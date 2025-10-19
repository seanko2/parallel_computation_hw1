#ifndef PARAMETERS_H
#define PARAMETERS_H

// Allow override from compiler flags
#ifndef PARAM_MC
#define PARAM_MC 256
#endif

#ifndef PARAM_NC
#define PARAM_NC 256
#endif

#ifndef PARAM_KC
#define PARAM_KC 128
#endif

#ifndef PARAM_MR
#define PARAM_MR 2
#endif

#ifndef PARAM_NR
#define PARAM_NR 30
#endif

inline constexpr int param_mc = PARAM_MC;
inline constexpr int param_nc = PARAM_NC;
inline constexpr int param_kc = PARAM_KC;
inline constexpr int param_mr = PARAM_MR;
inline constexpr int param_nr = PARAM_NR;

#endif // PARAMETERS_H
