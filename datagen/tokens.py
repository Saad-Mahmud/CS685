GLOBAL_TOKENS_MAP = { 0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
        10: '<prompt>', 11: '</prompt>\n', 12: '<reason>', 13: '</reason>\n', 14: '<ans>', 15: '</ans>',
        16: '#Answer only:', 17: '#Answer and Reason:', 18:'-', 19:'+', 20: '*', 21: 'mod', 22: "drop", 23:"<step>",
        24:"</step>\n", 25: "=", 26:"C:", 27:"A:", 28:' ', 31:'PAD', 30:'BOS', 29:'EOS'}

GLOBAL_TOKENS_RMAP = {v:k for (k,v) in GLOBAL_TOKENS_MAP.items()}