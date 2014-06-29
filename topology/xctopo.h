/* from Mike Stewart at NERSC:
    c12-5c2s7n1, which means the
    column 12, row 5, cage 2, slot 7 (blade), node 1.
*/

typedef struct xctopo_s
{
    int col;
    int row;
    int cage;
    int slot;
    int anode;
}
xctopo_t;
