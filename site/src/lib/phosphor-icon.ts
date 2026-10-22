/** Props for Phosphor web-component icons rendered as dynamic custom
 * elements. The package ships no typings; this signature keeps astro check
 * satisfied without changing runtime output. */
export type PhosphorIconProps = {
  size?: string;
  weight?: string;
  "aria-hidden"?: string | boolean;
} & Record<string, string | boolean | undefined>;
export type PhosphorIcon = (props: PhosphorIconProps) => unknown;
