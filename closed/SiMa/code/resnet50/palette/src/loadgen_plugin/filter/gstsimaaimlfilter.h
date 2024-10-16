/*
 * GStreamer
 * Copyright (C) 2023 SiMa.ai
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 * You should have received a copy of the GNU Library General Public
 * License along with this library; if not, write to the
 * Free Software Foundation, Inc., 59 Temple Place - Suite 330,
 * Boston, MA 02111-1307, USA.
 */

#ifndef __GST_ML_FILTER_H__
#define __GST_ML_FILTER_H__

#include <gst/gst.h>
#include <gst/base/gstbasetransform.h>
#include <stdio.h>

G_BEGIN_DECLS

#define GST_TYPE_ML_FILTER             (gst_ml_filter_get_type())
#define GST_ML_FILTER(obj)             (G_TYPE_CHECK_INSTANCE_CAST((obj),GST_TYPE_ML_FILTER,GstMlFilter))
#define GST_ML_FILTER_CLASS(klass)     (G_TYPE_CHECK_CLASS_CAST((klass),GST_TYPE_ML_FILTER,GstMlFilterClass))
#define GST_IS_ML_FILTER(obj)          (G_TYPE_CHECK_INSTANCE_TYPE((obj),GST_TYPE_ML_FILTER))
#define GST_IS_ML_FILTER_CLASS(klass)  (G_TYPE_CHECK_CLASS_TYPE((klass),GST_TYPE_ML_FILTER))

typedef struct _GstMlFilter      GstMlFilter;
typedef struct _GstMlFilterClass GstMlFilterClass;
typedef struct _GstMlFilterPrivate GstMlFilterPrivate;

struct _GstMlFilter {
  GstBaseTransform parent;
  GstMlFilterPrivate * priv;
  gboolean silent;
};

struct _GstMlFilterClass {
  GstBaseTransformClass parent_class;
};

GType gst_ml_filter_get_type (void);

G_END_DECLS

#endif /* __GST_ML_FILTER_H__ */
