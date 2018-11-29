function y = f_handleA(A,u,mode)

switch mode
    case 1
        y = A*u;
    case 2
        y = (u'*A)';
    otherwise
        error('Unknown mode passed to f_handleA in ftv_cs.m');
end

end