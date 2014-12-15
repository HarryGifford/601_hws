function r = isOctave ()
% from http://wiki.octave.org/Compatibility
persistent x;
if (isempty (x))
x = exist ('OCTAVE_VERSION', 'builtin');
end
r = x;
end
