const assert=require("assert"); const L=require("../static/kleinLayoutNG.js");
assert.deepStrictEqual(L.toBbox(0.5,0.1,0.1,0.9),[100,100,900,500]);
assert.strictEqual(L.compose("A cat",[]),"A cat");
const t=L.compose("A cat <cat_1> on a sofa.",[{id:"cat_1",bbox:[1,2,3,4],desc:"grey cat"}]);
assert.strictEqual(t,'A cat <cat_1> on a sofa. [{"id":"cat_1","bbox":[1,2,3,4],"desc":"grey cat"}]');
const u=L.compose("A room",[{id:"sofa",bbox:[0,0,10,10],desc:"red sofa, big"}]);
assert(u.startsWith("A room. red sofa <sofa>."),u);
assert.strictEqual(L.cleanId(" My Box! "),"my_box");
console.log("layout ok");
