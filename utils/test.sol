pragma solidity ^0.4.0;

contract shapeCalculator {
    function rectangle(uint w, uint h) returns (uint s, uint p) {
        s = w * h;
        p = 2 * (w + h);
    }
}
