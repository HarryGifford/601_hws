% minFunc
function [] = mexAll()

if ~is_octave()
% minFunc
fprintf('Compiling minFunc files...\n');
mex -outdir minFunc/compiled minFunc/mex/mcholC.c
mex -outdir minFunc/compiled minFunc/mex/lbfgsC.c
mex -outdir minFunc/compiled minFunc/mex/lbfgsAddC.c
mex -outdir minFunc/compiled minFunc/mex/lbfgsProdC.c

else
fprintf('Compiling minFunc files (octave version)...\n');
% working around the lack of an -outdir option in octave's mex

mexme('mcholC');
mexme('lbfgsC');
mexme('lbfgsAddC');
mexme('lbfgsProdC');
end

fprintf('Done.\n')
end

function mexme(fn)
cmd = sprintf('mkoctfile --mex --output minFunc/compiled/%s.mex minFunc/mex/%s.c', fn, fn) ;
status = system(cmd) ;
if status~=0
error('Executing command %s\n', cmd);
else
delete(sprintf('%s.o', fn));
fprintf('%s compiled\n', fn);
end

end

function r = is_octave ()
% from http://wiki.octave.org/Compatibility
   persistent x;
   if (isempty (x))
     x = exist ('OCTAVE_VERSION', 'builtin');
   end
   r = x;
 end